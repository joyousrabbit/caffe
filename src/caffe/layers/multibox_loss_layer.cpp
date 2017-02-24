#include <vector>

#include "caffe/layers/multibox_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {
	
template <typename Dtype>
void MultiboxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
    this->layer_param_.add_propagate_down(false);
  }
	
  BOX_K = 3;
  MAX_OBJ_CLASSES = 1;
}

template <typename Dtype>
void MultiboxLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
	
  CHECK_EQ(bottom[0]->height()*BOX_K, bottom[2]->height());
  CHECK_EQ(bottom[0]->channels(),(MAX_OBJ_CLASSES+4)*BOX_K);
  CHECK_EQ(bottom[1]->channels(),5);
  CHECK_EQ(bottom[2]->width(),4);
	
  num_priors_ = bottom[2]->height();

  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MultiboxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  const Dtype* prior_data = bottom[2]->cpu_data();
	
  Blob<Dtype> bestmatchBlob;
  bestmatchBlob.CopyFrom(*bottom[0],false,true);
	
  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes);
 
  for(int n=0;n<bottom[0]->num();n++){
//	  std::cout<<"Num: "<<n<<std::endl;
	  
	  //get gt
	  vector<NormalizedBBox> gt_bboxes;
  	  for(int h=0;h<bottom[1]->height();h++){
		  if(bottom[1]->data_at(n,4,h,0)<0){
			  break;
		  }
		  NormalizedBBox bbox;
		  bbox.set_xmin(bottom[1]->data_at(n,0,h,0));
		  bbox.set_ymin(bottom[1]->data_at(n,1,h,0));
		  bbox.set_xmax(bottom[1]->data_at(n,2,h,0));
		  bbox.set_ymax(bottom[1]->data_at(n,3,h,0));
		  bbox.set_label(bottom[1]->data_at(n,4,h,0));
		  float bbox_size = BBoxSize(bbox);
		  bbox.set_size(bbox_size);
		  gt_bboxes.push_back(bbox);
	  }
	  
	  //find best gt for each priorbox
	  Dtype overlap_threshold = 0.3;
	  vector<int> match_indices;
	  MatchBBox(gt_bboxes, prior_bboxes, overlap_threshold, &match_indices);
	  
	  for(int i=0;i<match_indices.size();i++){
		  int k = i%BOX_K;
		  int boxPos = i/BOX_K;
		  // reset classes to 0
		  int startC = k*(4+MAX_OBJ_CLASSES)+4;
		  for(int cla=0;cla<MAX_OBJ_CLASSES;cla++){
			  bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,startC+cla,boxPos,0)] = 0;
		  }
		  if(match_indices[i]<0){
			  //nothing to do
		  }else{
//			  std::cout<<"matched: "<<i<<"->"<<match_indices[i]<<std::endl;
			  //set bit of gt class to 1
			  int label = gt_bboxes[match_indices[i]].label();
			  int classBit = k*(4+MAX_OBJ_CLASSES)+4+label-1;
			  bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,classBit,boxPos,0)] = 1;
			  //set pos to gt's pos
			  NormalizedBBox encodedBox;
			  EncodeBBox(prior_bboxes[i],gt_bboxes[match_indices[i]],&encodedBox);
			  int posBit = k*(4+MAX_OBJ_CLASSES);
			  bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,posBit+0,boxPos,0)] = encodedBox.xmin();
			  bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,posBit+1,boxPos,0)] = encodedBox.ymin();
			  bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,posBit+2,boxPos,0)] = encodedBox.xmax();
			  bestmatchBlob.mutable_cpu_data()[bestmatchBlob.offset(n,posBit+3,boxPos,0)] = encodedBox.ymax();
		  }
	  }
  }
/*
  for(int n=0;n<bestmatchBlob.num();n++){
	  std::cout<<"Num: "<<n<<std::endl;
	  for(int h=0;h<bestmatchBlob.height();h++){
	  	for(int w=0;w<bestmatchBlob.width();w++){
	  		for(int c=0;c<bestmatchBlob.channels();c++){
				std::cout<<bestmatchBlob.data_at(n,c,h,w)<<",";
			}
			std::cout<<std::endl;
		}
	  }
  }
*/
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
void MultiboxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(MultiboxLossLayer);
#endif

INSTANTIATE_CLASS(MultiboxLossLayer);
REGISTER_LAYER_CLASS(MultiboxLoss);

}  // namespace caffe
