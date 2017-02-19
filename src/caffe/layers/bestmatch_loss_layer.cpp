#include <vector>

#include "caffe/layers/bestmatch_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BestmatchLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())<< "Inputs must have the same channnels.";
  CHECK_EQ(bottom[0]->height(),1)<<"Inputs[0] must has only one sample.";
  CHECK_EQ(bottom[0]->width(),1)<<"Inputs[0] must has only one sample.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void BestmatchLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Blob<Dtype> bestmatchBlob;
  bestmatchBlob.CopyFrom(*bottom[0],false,true);
  //get number of valid labels	
  for(int n=0;n<bottom[1]->num();n++){
//	  std::cout<<bottom[0]->data_at(n,0,0,0)<<","<<bottom[0]->data_at(n,1,0,0)<<","<<bottom[0]->data_at(n,2,0,0)<<","<<bottom[0]->data_at(n,3,0,0)<<","<<bottom[0]->data_at(n,4,0,0)<<std::endl;
	  int bestMatch = -1;
	  Dtype bestCost = 10000000;
	  for(int h=0;h<bottom[1]->height();h++){
		  if(bottom[1]->data_at(n,0,h,0)==0){
			  break;
		  }
		  Dtype cost = std::pow(bottom[1]->data_at(n,0,h,0)-bottom[0]->data_at(n,0,0,0),2)
			  		 + std::pow(bottom[1]->data_at(n,1,h,0)-bottom[0]->data_at(n,1,0,0),2)
			  		 + std::pow(bottom[1]->data_at(n,2,h,0)-bottom[0]->data_at(n,2,0,0),2)
			  		 + std::pow(bottom[1]->data_at(n,3,h,0)-bottom[0]->data_at(n,3,0,0),2);
		  if(cost<bestCost){
			  bestCost = cost;
			  bestMatch = h;
		  }
	  }
	  if(bestMatch<0){
	  	for(int c=4;c<bestmatchBlob.channels();c++){
			bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,c,0,0)] = 0;
		}
	  }else{
		for(int c=0;c<bestmatchBlob.channels();c++){
			bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,c,0,0)] = bottom[1]->data_at(n,c,bestMatch,0);
		}
	  }
//	  std::cout<<bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,0,0,0)]<<","<<bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,1,0,0)]<<","<<bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,2,0,0)]<<","<<bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,3,0,0)]<<","<<bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,4,0,0)]<<std::endl;
//	  std::cout<<bestCost<<std::endl;
//	  std::cout<<std::endl;
  }
	
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bestmatchBlob.cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void BestmatchLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BestmatchLossLayer);
#endif

INSTANTIATE_CLASS(BestmatchLossLayer);
REGISTER_LAYER_CLASS(BestmatchLoss);

}  // namespace caffe
