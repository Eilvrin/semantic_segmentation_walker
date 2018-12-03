#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


class CaffeWrapper {
 public:
  CaffeWrapper(){};
  void Initialize(const std::string& model_prototxt,
             const std::string& weights_caffemodel,
             const bool use_gpu);

  std::vector<float> Segment(const std::vector<float> &input_channels);

 private:
  std::shared_ptr<caffe::Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};
