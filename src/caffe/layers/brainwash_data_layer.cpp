#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/brainwash_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
BrainwashDataLayer<Dtype>::BrainwashDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param){
  loadDatums(param.data_param().source());
}

template <typename Dtype>
BrainwashDataLayer<Dtype>::~BrainwashDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void BrainwashDataLayer<Dtype>::loadDatums(const std::string& idlFilePath){
	datums_.clear();
	
	std::ifstream file(idlFilePath.c_str());
	CHECK(file.good())<<"File Error: "<<idlFilePath;
	string::size_type pos;
	pos = idlFilePath.rfind('/');
	folderPath_ = idlFilePath.substr(0,pos+1);
	
	string str;
	while(std::getline(file, str)) {
		pos = str.find('"',1);
		BrainwashDatum datum;
		datum.imgPath = str.substr(1,pos-0-1);
		string boxes = str.substr(pos+1,str.size()-1-pos-1-1);
		std::replace(boxes.begin(), boxes.end(), ':', ' ');
		std::replace(boxes.begin(), boxes.end(), '(', ' ');
		std::replace(boxes.begin(), boxes.end(), ')', ' ');
		std::replace(boxes.begin(), boxes.end(), ';', ' ');
		
		pos = boxes.find(',',0);
		if(pos != string::npos){
			std::stringstream ss(boxes);
			Box box;
			while( ss.good() ){
				string substr;
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
		}
		datums_.push_back(datum);
	}
	
	LOG(INFO) << "Database Nums: " << datums_.size();
}

template <typename Dtype>
void BrainwashDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  cv::Mat aImg = cv::imread(folderPath_+datums_[0].imgPath);
  std::vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(aImg.channels());
  top_shape.push_back(aImg.rows);
  top_shape.push_back(aImg.cols);
	
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
	vector<int> label_shape;
	label_shape.push_back(1);
	label_shape.push_back(4+MAX_OBJ_CLASSES); //px,py,pw,ph + classes
	label_shape.push_back(MAX_OBJ_NUMS);
	label_shape.push_back(1);
	
	this->transformed_label_.Reshape(label_shape);
	label_shape[0] = batch_size;
	top[1]->Reshape(label_shape);
	for (int i = 0; i < this->prefetch_.size(); ++i) {
	  this->prefetch_[i]->label_.Reshape(label_shape);
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
void BrainwashDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
	const BrainwashDatum& datum = datums_[rand()%datums_.size()];
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

	TransformDataLabel(datum, &(this->transformed_data_), &(this->transformed_label_));
    trans_time += timer.MicroSeconds();
  }
  
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void BrainwashDataLayer<Dtype>::TransformDataLabel(const BrainwashDatum& datum, Blob<Dtype>* data_blob, Blob<Dtype>* label_blob){
	//data
	cv::Mat aImg = cv::imread(folderPath_+datum.imgPath);
	this->data_transformer_->Transform(aImg, data_blob);
	//label
	CHECK_LE(datum.boxes.size(),label_blob->height());
	Dtype* label_data = label_blob->mutable_cpu_data();
	caffe_set(label_blob->count(), Dtype(0), label_data);
	Dtype N = std::max(aImg.rows,aImg.cols);
	for(int i=0;i<datum.boxes.size();i++){
		Dtype cx = (datum.boxes[i].x1+datum.boxes[i].x2)*0.5;
		Dtype cy = (datum.boxes[i].y1+datum.boxes[i].y2)*0.5;
		Dtype width = datum.boxes[i].x2-datum.boxes[i].x1;
		Dtype height = datum.boxes[i].y2-datum.boxes[i].y1;
		label_data[(0 * label_blob->height() + i) * label_blob->width()] = cx/N;
		label_data[(1 * label_blob->height() + i) * label_blob->width()] = cy/N;
		label_data[(2 * label_blob->height() + i) * label_blob->width()] = std::sqrt(width/N);
		label_data[(3 * label_blob->height() + i) * label_blob->width()] = std::sqrt(height/N);
		label_data[(4 * label_blob->height() + i) * label_blob->width()] = 1;
	}
}

INSTANTIATE_CLASS(BrainwashDataLayer);
REGISTER_LAYER_CLASS(BrainwashData);

}  // namespace caffe
