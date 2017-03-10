#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/cvppp_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/util/math_functions.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {

template <typename Dtype>
CvpppDataLayer<Dtype>::CvpppDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param){
  rows_ = 300;
  cols_ = 300;
  MAX_OBJ_NUMS = 50;
  MAX_OBJ_CLASSES = 1;
  loadDatums(param.data_param().source());
}

template <typename Dtype>
CvpppDataLayer<Dtype>::~CvpppDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void CvpppDataLayer<Dtype>::loadDatums(const std::string& filePath){
	datums_.clear();
	
	std::ifstream rootFile(filePath.c_str());
	CHECK(rootFile.good())<<"File Error: "<<filePath;
	string rootPath;
	while(std::getline(rootFile,rootPath)){
		CvpppDatum datum;
		datum.imgPath = rootPath;
		
		std::ifstream csvFile((rootPath+"_box.csv").c_str());
		string boxes;
		while(std::getline(csvFile,boxes)){
			std::stringstream ss(boxes);
			CvpppBox box;
			while( ss.good() ){
				string substr;
				getline( ss, substr, ',' );
				box.instanceID = std::atof(substr.c_str());
				getline( ss, substr, ',' );
				box.x1 = std::atof(substr.c_str());
				getline( ss, substr, ',' );
				box.y1 = std::atof(substr.c_str());
				getline( ss, substr, ',' );
				box.x2 = std::atof(substr.c_str());
				getline( ss, substr, ',' );
				box.y2 = std::atof(substr.c_str());
				box.classID = 1;
				datum.boxes.push_back(box);
			}
		}
		datums_.push_back(datum);
	}
	
	LOG(INFO) << "Database Nums: " << datums_.size();
}

template <typename Dtype>
void CvpppDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  std::vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(4);
  top_shape.push_back(rows_);
  top_shape.push_back(cols_);
	
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
	vector<int> label_shape;
	label_shape.push_back(1);
	label_shape.push_back(1); //px,py,pw,ph + classID
	label_shape.push_back(batch_size*MAX_OBJ_NUMS);
	label_shape.push_back(8);
	
	top[1]->Reshape(label_shape);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
	  this->prefetch_[i].label_.Reshape(label_shape);
	}

	/*
	for(int i=0;i<datums_[10].boxes.size();i++){
	cv::rectangle(aImg,cv::Point(datums_[10].boxes[i].x1, datums_[10].boxes[i].y1),cv::Point(datums_[10].boxes[i].x2, datums_[10].boxes[i].y2),cv::Scalar(255, 255, 255));
	}
	cv::imshow("debug",aImg);
	cv::waitKey();
	*/
}

// This function is called on prefetch thread
template<typename Dtype>
void CvpppDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

	vector<int> label_shape;
	label_shape.push_back(1);
	label_shape.push_back(4+1); //px,py,pw,ph + classID
	label_shape.push_back(MAX_OBJ_NUMS);
	label_shape.push_back(1);
	
	this->transformed_label_.Reshape(label_shape);
	
	label_shape[0] = batch_size;
	Blob<Dtype> tmp_label;
	tmp_label.Reshape(label_shape);

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    read_time += timer.MicroSeconds();

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    // Copy label.
	int offsetLabel = tmp_label.offset(item_id);
    Dtype* top_label = tmp_label.mutable_cpu_data();
	this->transformed_label_.set_cpu_data(top_label + offsetLabel);

	TransformDataLabel(&(this->transformed_data_), &(this->transformed_label_));
    trans_time += timer.MicroSeconds();
  }
  
  //convert label to ssd label format
  Dtype* ssd_label_data = batch->label_.mutable_cpu_data();
  caffe_set( batch->label_.count(), Dtype(-1), ssd_label_data);
  
  int idx = 0;

  for (int item_id = 0; item_id < tmp_label.num(); ++item_id) {
  	for(int instance_id = 0; instance_id < tmp_label.height(); ++ instance_id){
		if(tmp_label.data_at(item_id,4,instance_id,0)<0){
			break;
		}
  		ssd_label_data[idx++] = item_id;
  		ssd_label_data[idx++] = tmp_label.data_at(item_id,4,instance_id,0);
  		ssd_label_data[idx++] = instance_id;
  		ssd_label_data[idx++] = tmp_label.data_at(item_id,0,instance_id,0);
  		ssd_label_data[idx++] = tmp_label.data_at(item_id,1,instance_id,0);
  		ssd_label_data[idx++] = tmp_label.data_at(item_id,2,instance_id,0);
  		ssd_label_data[idx++] = tmp_label.data_at(item_id,3,instance_id,0);
  		ssd_label_data[idx++] = false;
  	}
  }
  
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void CvpppDataLayer<Dtype>::TransformDataLabel(Blob<Dtype>* data_blob, Blob<Dtype>* label_blob){
	const int ranIdx = rand()%datums_.size();
	CvpppDatum oriDatum = datums_[ranIdx];
	
	//data
	cv::Mat aImg = cv::imread(oriDatum.imgPath+"_rgb.png");
	cv::Mat labelImg = cv::imread(oriDatum.imgPath+"_label.bmp",0);
	CHECK_EQ(labelImg.channels(),1);
	
	cv::Mat memImg = cv::Mat::zeros(aImg.rows, aImg.cols, CV_8UC1);
	randomAssignDetected(oriDatum,labelImg,memImg);
	
//	cv::imshow("debug1",aImg);
//	cv::imshow("debug2",memImg);
//	cv::waitKey(5000);

	cv::resize(aImg, aImg, cv::Size(cols_,rows_));
	cv::resize(memImg, memImg, cv::Size(cols_,rows_),0,0, cv::INTER_NEAREST);
	memImg = memImg>100;
	
	std::vector<cv::Mat> channels;
	channels.push_back(aImg);
    channels.push_back(memImg);
	cv::Mat fin_img;
	cv::merge(channels, fin_img);
	
	
	CvpppDatum datum = oriDatum;
	
	const int flip = rand()%4;
	cv::Mat flip_img;
	switch(flip) {
		case 0 : flip_img = fin_img; //no flip
				 break;  
		case 1 : cv::flip(fin_img,flip_img,0); //by x-axis
				 datum = flipCvpppDatum(datum,0);
				 break;
		case 2 : cv::flip(fin_img,flip_img,1); //by y-axis
				 datum = flipCvpppDatum(datum,1);
				 break;
		case 3 : cv::flip(fin_img,flip_img,-1); //by x&y-axis
				 datum = flipCvpppDatum(datum,-1);
				 break;
	}
	
	const int rotate = rand()%2;
	cv::Mat rotate_img;
	switch(rotate){
		case 0: rotate_img = flip_img;
			    break;
		case 1: cv::transpose(flip_img,rotate_img);
				for(int i=0;i<datum.boxes.size();i++){
					float tmp = datum.boxes[i].y1;
					datum.boxes[i].y1 = datum.boxes[i].x1;
					datum.boxes[i].x1 = tmp;
					tmp = datum.boxes[i].y2;
					datum.boxes[i].y2 = datum.boxes[i].x2;
					datum.boxes[i].x2 = tmp;
				}
				break;
	}
	
	for(int i=0;i<datum.boxes.size();i++){
		float xMin = std::min(datum.boxes[i].x1,datum.boxes[i].x2);
		float xMax = std::max(datum.boxes[i].x1,datum.boxes[i].x2);
		float yMin = std::min(datum.boxes[i].y1,datum.boxes[i].y2);
		float yMax = std::max(datum.boxes[i].y1,datum.boxes[i].y2);
		datum.boxes[i].x1 = xMin;
		datum.boxes[i].x2 = xMax;
		datum.boxes[i].y1 = yMin;
		datum.boxes[i].y2 = yMax;
	}
	
	
	this->data_transformer_->Transform(rotate_img, data_blob);
	//label
	CHECK_LE(datum.boxes.size(),label_blob->height());
	Dtype* label_data = label_blob->mutable_cpu_data();
	caffe_set(label_blob->count(), Dtype(-1), label_data);
	
/*	std::cout<<oriDatum.imgPath<<std::endl;
	cv::Mat bgr( rotate_img.rows, rotate_img.cols, CV_8UC3 );
	cv::Mat alpha( rotate_img.rows, rotate_img.cols, CV_8UC1 );

	cv::Mat out[] = { bgr, alpha };
	int from_to[] = { 0,0, 1,1, 2,2, 3,3 };
	cv::mixChannels( &rotate_img, 1, out, 2, from_to, 4 );
	
	cv::Mat canvas = alpha.clone();
	cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
*/
	for(int i=0;i<datum.boxes.size();i++){
		label_data[(0 * label_blob->height() + i) * label_blob->width()] = datum.boxes[i].x1;
		label_data[(1 * label_blob->height() + i) * label_blob->width()] = datum.boxes[i].y1;
		label_data[(2 * label_blob->height() + i) * label_blob->width()] = datum.boxes[i].x2;
		label_data[(3 * label_blob->height() + i) * label_blob->width()] = datum.boxes[i].y2;
		label_data[(4 * label_blob->height() + i) * label_blob->width()] = 1;
		
//		int x1 = label_data[(0 * label_blob->height() + i) * label_blob->width()]*300;
//		int y1 = label_data[(1 * label_blob->height() + i) * label_blob->width()]*300;
//		int x2 = label_data[(2 * label_blob->height() + i) * label_blob->width()]*300;
//		int y2 = label_data[(3 * label_blob->height() + i) * label_blob->width()]*300;
		
//		cv::rectangle(canvas,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(255,0,255));
	}
	
//	cv::imshow("debug2",bgr);
//	cv::imshow("debug",canvas);
//	cv::waitKey(5000);
}

template<typename Dtype>
CvpppDatum CvpppDataLayer<Dtype>::flipCvpppDatum(const CvpppDatum& datum, int mode){
	CvpppDatum flippedDatum = datum;
	for(int i=0;i<flippedDatum.boxes.size();i++){
		switch(mode){
			case 0: flippedDatum.boxes[i].y1 = 1-flippedDatum.boxes[i].y1;
					flippedDatum.boxes[i].y2 = 1-flippedDatum.boxes[i].y2;
					break;
			case 1: flippedDatum.boxes[i].x1 = 1-flippedDatum.boxes[i].x1;
					flippedDatum.boxes[i].x2 = 1-flippedDatum.boxes[i].x2;
					break;
			case -1: flippedDatum.boxes[i].y1 = 1-flippedDatum.boxes[i].y1;
			        flippedDatum.boxes[i].y2 = 1-flippedDatum.boxes[i].y2;
			        flippedDatum.boxes[i].x1 = 1-flippedDatum.boxes[i].x1;
					flippedDatum.boxes[i].x2 = 1-flippedDatum.boxes[i].x2;
					break;
		}
	}
	return flippedDatum;
}

template<typename Dtype>
void CvpppDataLayer<Dtype>::randomAssignDetected(CvpppDatum& datum,const cv::Mat& labelImg,cv::Mat& memImg){
	//at least one instance rest
	int luckyNum = rand()%datum.boxes.size();
	std::vector<CvpppBox> unDetectedBoxes;
	std::set<int> detectedSet;
	for(int i=0;i<datum.boxes.size();i++){
		if(i==luckyNum || rand()%2==1){
			unDetectedBoxes.push_back(datum.boxes[i]);
		}else{
			detectedSet.insert(datum.boxes[i].instanceID);
		}
	}
	
	CHECK_EQ(labelImg.rows,memImg.rows);
	CHECK_EQ(labelImg.cols,memImg.cols);
	
	for(int y=0;y<labelImg.rows;y++){
		for(int x=0;x<labelImg.cols;x++){
			const bool is_in = detectedSet.find(labelImg.at<uchar>(y,x)) != detectedSet.end();
			if(is_in){
				memImg.at<uchar>(y,x) = 255;
			}
		}
	}
	
	datum.boxes = unDetectedBoxes;
}

INSTANTIATE_CLASS(CvpppDataLayer);
REGISTER_LAYER_CLASS(CvpppData);

}  // namespace caffe
