#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/blackpoint_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
BlackpointDataLayer<Dtype>::BlackpointDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param){
  rows_ = 160;
  cols_ = 160;
}

template <typename Dtype>
BlackpointDataLayer<Dtype>::~BlackpointDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void BlackpointDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  std::vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(1);
  top_shape.push_back(rows_);
  top_shape.push_back(cols_);
	
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
}

// This function is called on prefetch thread
template<typename Dtype>
void BlackpointDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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

	GenerateDataLabel(&(this->transformed_data_), &(this->transformed_label_));
    trans_time += timer.MicroSeconds();
  }
  
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void BlackpointDataLayer<Dtype>::GenerateDataLabel(Blob<Dtype>* data_blob, Blob<Dtype>* label_blob){	
	int x1 = rand()%(cols_-10);
	int y1 = rand()%(rows_-10);
	int d = rand()%std::min(rows_-y1,cols_-x1);
	d = std::max(d,10);
	int x2 = x1+d;
	int y2 = y1+d;
	
	Dtype cx = (x1+x2)*0.5;
	Dtype cy = (y1+y2)*0.5;
	Dtype width = x2-x1;
	Dtype height = y2-y1;
	
	//data
	cv::Mat aImg = cv::Mat::zeros(rows_, cols_, CV_8U);
	this->data_transformer_->Transform(aImg, data_blob);
	//label
	Dtype* label_data = label_blob->mutable_cpu_data();
	caffe_set(label_blob->count(), Dtype(0), label_data);
	Dtype N = std::max(aImg.rows,aImg.cols);
	
	cv::circle(aImg,cv::Point(cx,cy),d*0.5,cv::Scalar(255),-1);
	
	int i = 0;
	label_data[(0 * label_blob->height() + i) * label_blob->width()] = cx/N;
	label_data[(1 * label_blob->height() + i) * label_blob->width()] = cy/N;
	label_data[(2 * label_blob->height() + i) * label_blob->width()] = std::sqrt(width/N);
	label_data[(3 * label_blob->height() + i) * label_blob->width()] = std::sqrt(height/N);
	label_data[(4 * label_blob->height() + i) * label_blob->width()] = 1;
	
//	std::cout<<label_data[0]<<","<<label_data[1]<<","<<label_data[2]<<","<<label_data[3]<<","<<label_data[4]<<std::endl;
//	cv::imshow("debug",aImg);
//	cv::waitKey(10000);
}

INSTANTIATE_CLASS(BlackpointDataLayer);
REGISTER_LAYER_CLASS(BlackpointData);

}  // namespace caffe
