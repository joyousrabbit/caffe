#ifndef CAFFE_BITS_DATA_LAYER_HPP_
#define CAFFE_BITS_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class BitsDataLayer : public BasePrefetchingDataLayer<Dtype> {
 /* This data layer is reserved for lstm test, because it's TxNx... order.
 */
 public:
  explicit BitsDataLayer(const LayerParameter& param);
  virtual ~BitsDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "BitsData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
  std::vector<int> convertInt2Bits(int x, int len);
  int bitsNum_;
};

}  // namespace caffe

#endif  // CAFFE_BITS_DATA_LAYER_HPP_
