#include "semantic_seg/CaffeWrapper.h"

using namespace caffe;
using std::string;

void CaffeWrapper::Initialize(const string& model_prototxt,
                           const string& weights_caffemodel,
                           const bool use_gpu) {

  if (use_gpu){
    Caffe::set_mode(Caffe::GPU);
  } else {
    Caffe::set_mode(Caffe::CPU);
  }

  /* Load the network. */
  net_.reset(new Net<float>(model_prototxt, TEST));
  net_->CopyTrainedLayersFrom(weights_caffemodel);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 4)
    << "Input layer should have 4 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

}


std::vector<float> CaffeWrapper::Segment(const std::vector<float> &input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  float* input_data = input_layer->mutable_cpu_data();
  std::memcpy(input_data, input_channels.data(), sizeof(float)*input_channels.size());

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + input_geometry_.height*input_geometry_.width*output_layer->channels();
  return std::vector<float>(begin, end);
}

