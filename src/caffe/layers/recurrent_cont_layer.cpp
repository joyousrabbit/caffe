#include <vector>

#include "caffe/layers/recurrent_cont_layer.hpp"

namespace caffe {

template <typename Dtype>
void RecurrentContLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RecurrentContLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	CHECK_GE(bottom[0]->num_axes(), 3);
	vector<int> top_shape;
	top_shape.push_back(bottom[0]->num());
	top_shape.push_back(bottom[0]->channels());
	top[0]->Reshape(top_shape);
}
	
template <typename Dtype>
void RecurrentContLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
	caffe_set(top[0]->count(), Dtype(1), top[0]->mutable_cpu_data());
	int t = 0;
	for(int n=0;n<top[0]->channels();n++){
		top[0]->mutable_cpu_data()[top[0]->offset(t,n,0,0)] = 0;
	}
}

INSTANTIATE_CLASS(RecurrentContLayer);
REGISTER_LAYER_CLASS(RecurrentCont);

}  // namespace caffe
