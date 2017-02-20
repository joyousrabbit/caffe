#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/bits_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
BitsDataLayer<Dtype>::BitsDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param){
  bitsNum_ = 8;
}

template <typename Dtype>
BitsDataLayer<Dtype>::~BitsDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void BitsDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  std::vector<int> top_shape;
  top_shape.push_back(bitsNum_);
  top_shape.push_back(batch_size);
  top_shape.push_back(2);
  top_shape.push_back(1);
	
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
	label_shape.push_back(bitsNum_);
	label_shape.push_back(batch_size);
	label_shape.push_back(1);
	label_shape.push_back(1);
	
	top[1]->Reshape(label_shape);
	for (int i = 0; i < this->prefetch_.size(); ++i) {
	  this->prefetch_[i]->label_.Reshape(label_shape);
	}
}

// This function is called on prefetch thread
template<typename Dtype>
void BitsDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  
  
  
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    read_time += timer.MicroSeconds();

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
	
	int maxNum = std::pow(2,bitsNum_);
	int num1 = std::rand()%(maxNum/2-1);
	int num2 = std::rand()%(maxNum/2-1);
	int sum = num1+num2;
	
	std::vector<int> bits1 = convertInt2Bits(num1,bitsNum_);
	std::vector<int> bits2 = convertInt2Bits(num2,bitsNum_);
	std::vector<int> bitsSum = convertInt2Bits(sum,bitsNum_);
	
	for(int t=0;t<bitsNum_;t++){
		batch->data_.mutable_cpu_data()[batch->data_.offset(t,item_id,0,0)] = bits1[t];
		batch->data_.mutable_cpu_data()[batch->data_.offset(t,item_id,1,0)] = bits2[t];
		batch->label_.mutable_cpu_data()[batch->label_.offset(t,item_id,0,0)] = bitsSum[t];
	}
	
    trans_time += timer.MicroSeconds();
  }
  
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
std::vector<int> BitsDataLayer<Dtype>::convertInt2Bits(int x, int len) {
  vector<int> ret;
  while(x) {
    if (x&1)
      ret.push_back(1);
    else
      ret.push_back(0);
    x>>=1;  
  }
  for(int i=ret.size();i<len;i++){
  	ret.push_back(0);
  }
  return ret;
}

INSTANTIATE_CLASS(BitsDataLayer);
REGISTER_LAYER_CLASS(BitsData);

}  // namespace caffe
