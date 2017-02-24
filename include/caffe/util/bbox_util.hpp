#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#ifndef CAFFE_UTIL_BBOX_UTIL_H_
#define CAFFE_UTIL_BBOX_UTIL_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

namespace caffe {
void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox);
float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);
	
// Compute bbox size.
float BBoxSize(const NormalizedBBox& bbox);	
	
// Get prior bounding boxes from prior_data.
//    prior_data: 1 x 2 x num_priors * 4 x 1 blob.
//    num_priors: number of priors.
//    prior_bboxes: stores all the prior bboxes in the format of NormalizedBBox.
//    prior_variances: stores all the variances needed by prior bboxes.
template <typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes);

void MatchBBox(const vector<NormalizedBBox>& gt_bboxes,
    const vector<NormalizedBBox>& pred_bboxes, const float overlap_threshold,
    vector<int>* match_indices);
	
void EncodeBBox(
    const NormalizedBBox& prior_bbox, const NormalizedBBox& bbox, NormalizedBBox* encode_bbox);

}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
