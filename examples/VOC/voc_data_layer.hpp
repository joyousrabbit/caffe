#ifndef CAFFE_VOC_DATA_LAYER_HPP_
#define CAFFE_VOC_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	
  struct VocBox {
  	float x1,y1,x2,y2;
	int classID;
	int instanceID;
  };
	
  struct VocDatum { 
	string imgPath;
	std::vector<VocBox> boxes;
  };

template <typename Dtype>
class VocDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VocDataLayer(const LayerParameter& param);
  virtual ~VocDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "VocData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  
  virtual void load_batch(Batch<Dtype>* batch);
  void loadDatums(const std::string& idlFilePath);
  void TransformDataLabel(Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label); 
  VocDatum flipVocDatum(const VocDatum& datum, int mode);
  void randomAssignDetected(VocDatum& datum,const cv::Mat& labelImg,cv::Mat& memImg);

  std::vector<VocDatum> datums_;
  int rows_;
  int cols_;
  int MAX_OBJ_NUMS;
  string rootPath;
  
  Blob<Dtype> transformed_label_;
};

}  // namespace caffe

#endif  // CAFFE_VOC_DATA_LAYER_HPP_
