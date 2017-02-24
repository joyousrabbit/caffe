#ifndef CAFFE_MULTIBOX_LOSS_LAYER_HPP_
#define CAFFE_MULTIBOX_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
template <typename Dtype>
class MultiboxLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiboxLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiboxLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }

 protected:
  /// @copydoc MultiboxLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  int BOX_K;
  int MAX_OBJ_CLASSES;
  int num_priors_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
