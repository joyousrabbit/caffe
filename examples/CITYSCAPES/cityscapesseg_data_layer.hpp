#ifndef CAFFE_CITYSCAPESSEG_DATA_LAYER_HPP_
#define CAFFE_CITYSCAPESSEG_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	
  struct CityscapesSegBox {
  	float x1,y1,x2,y2;
	int classID;
	int instanceID;
  };
	
  struct CityscapesSegDatum { 
	string imgPath;
	std::vector<CityscapesSegBox> boxes;
  };

template <typename Dtype>
class CityscapesSegDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit CityscapesSegDataLayer(const LayerParameter& param);
  virtual ~CityscapesSegDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "CityscapesSegData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  
  virtual void load_batch(Batch<Dtype>* batch);
  void loadDatums(const std::string& idlFilePath);
  void TransformDataLabel(Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label); 
  CityscapesSegDatum flipCityscapesSegDatum(const CityscapesSegDatum& datum, int mode);
  void randomAssignDetected(CityscapesSegDatum& datum,const cv::Mat& labelImg,cv::Mat& memImg);
  int getDominantInstanceID(const cv::Mat& labelImg);

  std::vector<CityscapesSegDatum> datums_;
  int rows_;
  int cols_;
  
  Blob<Dtype> transformed_label_;
};

}  // namespace caffe

#endif  // CAFFE_CITYSCAPESSEG_DATA_LAYER_HPP_
