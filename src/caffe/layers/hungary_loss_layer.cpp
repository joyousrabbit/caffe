#include <vector>

#include "caffe/layers/hungary_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/hungarian.hpp"

namespace caffe {

template <typename Dtype>
void HungaryLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())<< "Inputs must have the same channnels.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void HungaryLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Blob<Dtype> bestmatchBlob;
  bestmatchBlob.CopyFrom(*bottom[0],false,true);
	
  //get number of valid labels	
  for(int n=0;n<bottom[1]->num();n++){
	  const int kMaxNumPred = 20;
	  CHECK_LE(bottom[0]->height(), kMaxNumPred);
	  double match_cost[kMaxNumPred * kMaxNumPred];
	  
	  for(int b0=0;b0<bottom[0]->height();b0++){
	  	for(int b1=0;b1<bottom[1]->height();b1++){
			Dtype cost = std::pow(bottom[1]->data_at(n,0,b1,0)-bottom[0]->data_at(n,0,b0,0),2)
			  		 + std::pow(bottom[1]->data_at(n,1,b1,0)-bottom[0]->data_at(n,1,b0,0),2)
			  		 + std::pow(bottom[1]->data_at(n,2,b1,0)-bottom[0]->data_at(n,2,b0,0),2)
			  		 + std::pow(bottom[1]->data_at(n,3,b1,0)-bottom[0]->data_at(n,3,b0,0),2);
			match_cost[b0*bottom[1]->height()+b1] = cost;
//			std::cout<<cost<<" ";
		}
//		std::cout<<std::endl;
	  }
	  
	  hungarian_problem_t p;
      double** m = array_to_matrix(match_cost, bottom[0]->height(), bottom[1]->height());
      hungarian_init(&p, m, bottom[0]->height(), bottom[1]->height(), HUNGARIAN_MODE_MINIMIZE_COST);
      hungarian_solve(&p);
	  
	  std::vector<int> assignment(bottom[0]->height(),-1);
	  for (int i = 0; i < bottom[0]->height(); ++i) {
        for (int j = 0; j < bottom[1]->height(); ++j) {
          if (p.assignment[i][j] == HUNGARIAN_ASSIGNED) {
            assignment[i] = j;
          }
        }
      }
	  
	  hungarian_free(&p);
      for (int i = 0; i < bottom[0]->height(); ++i) {
        free(m[i]);
      }
      free(m);
	  
	  for(int i=0;i<assignment.size();i++){
//		  std::cout<<assignment[i]<<std::endl;
		  if(assignment[i]<0){
			for(int c=4;c<bestmatchBlob.channels();c++){
				bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,c,i,0)] = 0;
			}
		  }else{
			for(int c=0;c<bestmatchBlob.channels();c++){
				bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,c,i,0)] = bottom[1]->data_at(n,c,assignment[i],0);
			}			  
		  }
	  }
//	  std::cout<<std::endl;

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
void HungaryLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(HungaryLossLayer);
#endif

INSTANTIATE_CLASS(HungaryLossLayer);
REGISTER_LAYER_CLASS(HungaryLoss);

}  // namespace caffe
