#ifndef CAFFE_BLACKPOINT_DATA_LAYER_HPP_
#define CAFFE_BLACKPOINT_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#define MAX_OBJ_NUMS 5
#define MAX_OBJ_CLASSES 1

namespace caffe {

template <typename Dtype>
class BlackpointDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit BlackpointDataLayer(const LayerParameter& param);
  virtual ~BlackpointDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "BlackpointData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  struct Box {
  	Dtype x1,y1,x2,y2;
	int classID;
  };
  
  virtual void load_batch(Batch<Dtype>* batch);
  void GenerateDataLabel(Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label); 
  
  Blob<Dtype> transformed_label_;
  int rows_;
  int cols_;
};

}  // namespace caffe

#endif  // CAFFE_BLACKPOINT_DATA_LAYER_HPP_
