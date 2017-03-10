#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/cityscapesseg_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/util/math_functions.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {

template <typename Dtype>
CityscapesSegDataLayer<Dtype>::CityscapesSegDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param){
  rows_ = 48;
  cols_ = 48;
  loadDatums(param.data_param().source());
}

template <typename Dtype>
CityscapesSegDataLayer<Dtype>::~CityscapesSegDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void CityscapesSegDataLayer<Dtype>::loadDatums(const std::string& filePath){
	datums_.clear();
	
	std::ifstream rootFile(filePath.c_str());
	CHECK(rootFile.good())<<"File Error: "<<filePath;
	string rootPath;
	while(std::getline(rootFile,rootPath)){
		CityscapesSegDatum datum;
		datum.imgPath = rootPath;
		
		std::ifstream csvFile((rootPath+"_gtFine_box.csv").c_str());
		string boxes;
		while(std::getline(csvFile,boxes)){
			std::stringstream ss(boxes);
			CityscapesSegBox box;
			while( ss.good() ){
				string substr;
				getline( ss, substr, ',' );
				box.instanceID = std::atof(substr.c_str());
				getline( ss, substr, ',' );
				box.classID = std::atof(substr.c_str());;
				getline( ss, substr, ',' );
				box.x1 = std::atof(substr.c_str());
				getline( ss, substr, ',' );
				box.y1 = std::atof(substr.c_str());
				getline( ss, substr, ',' );
				box.x2 = std::atof(substr.c_str());
				getline( ss, substr, ',' );
				box.y2 = std::atof(substr.c_str());
				datum.boxes.push_back(box);
			}
			CHECK_LT(box.x1,box.x2)<<rootPath;
			CHECK_LT(box.y1,box.y2)<<rootPath;
		}
		datums_.push_back(datum);
	}
	
	LOG(INFO) << "Database Nums: " << datums_.size();
}

template <typename Dtype>
void CityscapesSegDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
	label_shape.push_back(1);
	label_shape.push_back(rows_);
	label_shape.push_back(cols_);
	
	 this->transformed_label_.Reshape(label_shape);
	label_shape[0] = batch_size;
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
void CityscapesSegDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    read_time += timer.MicroSeconds();

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    // Copy label.
	int offsetLabel = batch->label_.offset(item_id);
    Dtype* top_label = batch->label_.mutable_cpu_data();
	this->transformed_label_.set_cpu_data(top_label + offsetLabel);

	TransformDataLabel(&(this->transformed_data_), &(this->transformed_label_));
    trans_time += timer.MicroSeconds();
  }
  
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void CityscapesSegDataLayer<Dtype>::TransformDataLabel(Blob<Dtype>* data_blob, Blob<Dtype>* label_blob){
	CityscapesSegDatum oriDatum;
	while(oriDatum.boxes.size()==0){
		oriDatum = datums_[rand()%datums_.size()];
	}
	
	//data
	cv::Mat aImg = cv::imread(oriDatum.imgPath+"_leftImg8bit.small.png");
	cv::Mat labelImg = cv::imread(oriDatum.imgPath+"_gtFine_instanceIds.small.png",CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	CHECK_EQ(labelImg.type(),CV_16UC1);
	
	cv::Mat memImg = cv::Mat::zeros(aImg.rows, aImg.cols, CV_8UC1);
	randomAssignDetected(oriDatum,labelImg,memImg);
	
	cv::Mat mask = memImg == 255;
	labelImg.setTo(0, mask);
	
//	cv::imshow("debug1",aImg);
//	cv::imshow("debug2",labelImg>0);
//	cv::waitKey(3000);
	
	std::vector<cv::Mat> channels;
	channels.push_back(aImg);
    channels.push_back(memImg);
	cv::Mat fin_img;
	cv::merge(channels, fin_img);
	
	CityscapesSegDatum datum = oriDatum;
	
	//	const int flip = rand()%4;
	const int flip = rand()%2*2; //no flip or by y
	cv::Mat flip_img,flip_labelImg;
	switch(flip) {
		case 0 : flip_img = fin_img; //no flip
				 flip_labelImg = labelImg;
				 break;  
		case 1 : cv::flip(fin_img,flip_img,0); //by x-axis
		 		 cv::flip(labelImg,flip_labelImg,0);
				 datum = flipCityscapesSegDatum(datum,0);
				 break;
		case 2 : cv::flip(fin_img,flip_img,1); //by y-axis
				 cv::flip(labelImg,flip_labelImg,1);
				 datum = flipCityscapesSegDatum(datum,1);
				 break;
		case 3 : cv::flip(fin_img,flip_img,-1); //by x&y-axis
				 cv::flip(labelImg,flip_labelImg,-1);
				 datum = flipCityscapesSegDatum(datum,-1);
				 break;
	}
	
	//	const int rotate = rand()%2;
	const int rotate = 0; //no rotate
	cv::Mat rotate_img,rotate_labelImg;
	switch(rotate){
		case 0: rotate_img = flip_img;
				rotate_labelImg = flip_labelImg;
			    break;
		case 1: cv::transpose(flip_img,rotate_img);
				cv::transpose(flip_labelImg,rotate_labelImg);
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
	
	CHECK_GT(datum.boxes.size(),0);
	const int ranObjInd = rand()%datum.boxes.size();
	
	float pcx = (datum.boxes[ranObjInd].x1+datum.boxes[ranObjInd].x2)*0.5;
	float pcy = (datum.boxes[ranObjInd].y1+datum.boxes[ranObjInd].y2)*0.5;
	float pw = datum.boxes[ranObjInd].x2-datum.boxes[ranObjInd].x1;
	float ph = datum.boxes[ranObjInd].y2-datum.boxes[ranObjInd].y1;
	
//	float maxDriftX = 0.2*pw;
//	float maxDriftY = 0.2*ph;
//	pcx += (rand()%2*2-1)*maxDriftX*(rand()%10)/10.0;
//	pcy += (rand()%2*2-1)*maxDriftY*(rand()%10)/10.0;
	
	pw *= 0.8+(rand()%10)/10.0;
	ph *= 0.8+(rand()%10)/10.0;
	
	double x1 = pcx-pw*0.5;
	double y1 = pcy-ph*0.5;
	double x2 = pcx+pw*0.5;
	double y2 = pcy+ph*0.5;
	
	x1 = std::max(x1,0.0);
	y1 = std::max(y1,0.0);
	x2 = std::min(x2,1.0);
	y2 = std::min(y2,1.0);
	
	CHECK_EQ(rotate_img.rows,rotate_labelImg.rows);
	CHECK_EQ(rotate_img.cols,rotate_labelImg.cols);
	
	y1 = round(y1*rotate_img.rows);
	y2 = round(y2*rotate_img.rows);
	x1 = round(x1*rotate_img.cols);
	x2 = round(x2*rotate_img.cols);
	CHECK_GT(y2,y1);
	CHECK_GT(x2,x1);
	
	cv::Mat segImg(rotate_img,cv::Range( y1, y2), cv::Range( x1,x2 ));
	cv::Mat segLabelImg(rotate_labelImg,cv::Range( y1, y2), cv::Range( x1,x2 ));
//	cv::Mat label = segLabelImg==datum.boxes[ranObjInd].instanceID;
	int dominantInstanceID = getDominantInstanceID(segLabelImg);
	CHECK_GT(dominantInstanceID,0);
	cv::Mat label = segLabelImg==dominantInstanceID;
	
	CHECK_GT(segImg.rows*segImg.cols,0)<<oriDatum.imgPath<<":"<<datum.boxes[ranObjInd].instanceID<<" ["<<x1<<","<<y1<<","<<x2<<","<<y2<<"]";
	
	cv::resize(segImg, segImg, cv::Size(cols_,rows_));
	cv::resize(label,label,cv::Size(cols_,rows_),0,0,cv::INTER_NEAREST);
	
	this->data_transformer_->Transform(segImg, data_blob);
	this->data_transformer_->Transform(label, label_blob);
	
/*	if(dominantInstanceID==0){	
//	if(true){	
		std::cout<<oriDatum.imgPath<<std::endl;
		cv::Mat bgr( segImg.rows, segImg.cols, CV_8UC3 );
		cv::Mat alpha( segImg.rows, segImg.cols, CV_8UC1 );

		cv::Mat out[] = { bgr, alpha };
		int from_to[] = { 0,0, 1,1, 2,2, 3,3 };
		cv::mixChannels( &segImg, 1, out, 2, from_to, 4 );

		cv::Mat canvas = bgr.clone();

		cv::imshow("debug2",canvas);
//		cv::imshow("debug1",segLabelImg==dominantInstanceID);
		cv::imshow("debug",label);
		cv::waitKey(0);
	}
	*/
}

template<typename Dtype>
CityscapesSegDatum CityscapesSegDataLayer<Dtype>::flipCityscapesSegDatum(const CityscapesSegDatum& datum, int mode){
	CityscapesSegDatum flippedDatum = datum;
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
void CityscapesSegDataLayer<Dtype>::randomAssignDetected(CityscapesSegDatum& datum,const cv::Mat& labelImg,cv::Mat& memImg){
	//at least one instance rest
	if(datum.boxes.size()==0){
		return;
	}
	
	int luckyNum = rand()%datum.boxes.size();
	std::vector<CityscapesSegBox> unDetectedBoxes;
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
			const bool is_in = detectedSet.find(labelImg.at<unsigned short>(y,x)) != detectedSet.end();
			if(is_in){
				memImg.at<uchar>(y,x) = 255;
			}
		}
	}
	
	datum.boxes = unDetectedBoxes;
}

template<typename Dtype>
int CityscapesSegDataLayer<Dtype>::getDominantInstanceID(const cv::Mat& labelImg){
	int fre[40000] = {};
	for(int y=0;y<labelImg.rows;y++){
		for(int x=0;x<labelImg.cols;x++){
			int aLabel = labelImg.at<unsigned short>(y,x);
			CHECK_LT(aLabel,40000)<<"sorry I can't handle label bigger than 40000";
			int classID = aLabel;
			if(classID>1000){
				classID = classID/1000;
			}
			if(aLabel>0 && (classID==24 || classID==25 || classID==26 || classID==27 || classID==28 || classID==31 || classID==32 || classID==33)){
				fre[aLabel]++;
			}
		}
	}
	int ind = 0;
	for(int i=0;i<40000;i++){
		if(fre[i]>fre[ind]){
			ind = i;
		}
	}
	if(ind==0) ind=50000;
	return ind;
}
	
INSTANTIATE_CLASS(CityscapesSegDataLayer);
REGISTER_LAYER_CLASS(CityscapesSegData);

}  // namespace caffe
