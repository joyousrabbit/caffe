#include <algorithm>
#include <csignal>
#include <ctime>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/iterator/counting_iterator.hpp"

#include "caffe/util/bbox_util.hpp"

namespace caffe {
	
void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox) {
  if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
      bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin()) {
    // Return [0, 0, 0, 0] if there is no intersection.
    intersect_bbox->set_xmin(0);
    intersect_bbox->set_ymin(0);
    intersect_bbox->set_xmax(0);
    intersect_bbox->set_ymax(0);
  } else {
    intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
    intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
    intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
    intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
  }
}
	
float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
  NormalizedBBox intersect_bbox;
  IntersectBBox(bbox1, bbox2, &intersect_bbox);
  float intersect_width, intersect_height;

  intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
  intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
 
  if (intersect_width > 0 && intersect_height > 0) {
    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = BBoxSize(bbox1);
    float bbox2_size = BBoxSize(bbox2);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
  } else {
    return 0.;
  }
}

float BBoxSize(const NormalizedBBox& bbox) {
  if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin()) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0;
  } else {
    if (bbox.has_size()) {
      return bbox.size();
    } else {
      float width = bbox.xmax() - bbox.xmin();
      float height = bbox.ymax() - bbox.ymin();
      return width * height;
    }
  }
}

template <typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes) {
  prior_bboxes->clear();
  for (int i = 0; i < num_priors; ++i) {
    int start_idx = i * 4;
    NormalizedBBox bbox;
    bbox.set_xmin(prior_data[start_idx]);
    bbox.set_ymin(prior_data[start_idx + 1]);
    bbox.set_xmax(prior_data[start_idx + 2]);
    bbox.set_ymax(prior_data[start_idx + 3]);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    prior_bboxes->push_back(bbox);
  }
}
	
// Explicit initialization.
template void GetPriorBBoxes(const float* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes);
template void GetPriorBBoxes(const double* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes);
	
void MatchBBox(const vector<NormalizedBBox>& gt_bboxes,
    const vector<NormalizedBBox>& pred_bboxes, const float overlap_threshold,
    vector<int>* match_indices) {
	int num_pred = pred_bboxes.size();
	match_indices->clear();
    match_indices->resize(num_pred, -1);
	
	for(int i=0;i<num_pred;i++){
		float maxOverlap = -1;
		for(int j=0;j<gt_bboxes.size();j++){
			float jac = JaccardOverlap(pred_bboxes[i],gt_bboxes[j]);
			if(jac>overlap_threshold && jac>maxOverlap){
				maxOverlap = jac;
				(*match_indices)[i] = j;
			}
		}
	}
}
	
void EncodeBBox(
    const NormalizedBBox& prior_bbox, const NormalizedBBox& bbox, NormalizedBBox* encode_bbox) {
  
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
    float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

    float bbox_width = bbox.xmax() - bbox.xmin();
    CHECK_GT(bbox_width, 0);
    float bbox_height = bbox.ymax() - bbox.ymin();
    CHECK_GT(bbox_height, 0);
    float bbox_center_x = (bbox.xmin() + bbox.xmax()) / 2.;
    float bbox_center_y = (bbox.ymin() + bbox.ymax()) / 2.;

    encode_bbox->set_xmin((bbox_center_x - prior_center_x) / prior_width);
    encode_bbox->set_ymin((bbox_center_y - prior_center_y) / prior_height);
    encode_bbox->set_xmax(log(bbox_width / prior_width));
    encode_bbox->set_ymax(log(bbox_height / prior_height));
}

}  // namespace caffe
