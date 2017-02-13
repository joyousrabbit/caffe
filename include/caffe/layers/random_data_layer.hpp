#ifndef CAFFE_RANDOM_DATA_LAYER_HPP_
#define CAFFE_RANDOM_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class RandomDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit RandomDataLayer(const LayerParameter& param);
  virtual ~RandomDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "RandomData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
  void getAllKeys();

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
  std::vector<string> keys_;
};

}  // namespace caffe

#endif  // CAFFE_RANDOM_DATA_LAYER_HPP_
