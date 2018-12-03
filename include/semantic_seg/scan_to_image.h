#ifndef SCAN_TO_IMAGE_H
#define SCAN_TO_IMAGE_H

#include <string>
#include <vector>
#include <iostream>
#include <ctime>
#include <chrono>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/pcd_io.h>

#include <ros/ros.h>
#include <tf/transform_listener.h>

#include "semantic_seg/CaffeWrapper.h"

namespace semantic_seg {

// Max range for Velodyne
static const float MAX_RANGE_ = 100.0f;

class ScanToImage
{
public:

  typedef pcl::PointXYZINormal PclPoint;
  typedef pcl::PointCloud<PclPoint> PclCloud;
  typedef PclCloud::Ptr PclCloudPtr;

  ScanToImage(ros::NodeHandle node, ros::NodeHandle private_nh);  
  ~ScanToImage(){};

private:
  void processScan(const PclCloudPtr &point_cloud);
  Eigen::Vector2i getProjectedIndexVelodyne(const Eigen::Vector3f& point);
  float normalizeAngle(float angle);

  ros::Subscriber assembled_cloud_;

  int width_;
  int height_;

  // Parameters for spherical projection
  float xAngularResolution_;
  float xAngleOffset_;
  float yAngleOffset_;
  float yAngularResolution_;
  float xAngleRange_;

  std::vector<float> depth_;
  std::vector<float> normals_xyz_;
  std::vector<float> idx_;

  tf::TransformListener listener_;
  std::string target_frame_id_;
  ros::Publisher labeled_cloud_pub_;
  ros::Publisher unlabeled_cloud_pub_;

  std::string model_prototxt_;
  std::string weights_caffemodel_;
  bool use_gpu_;
  int num_classes_;

  CaffeWrapper caffe_wrapper;
};

} //namespace semantic_seg


#endif // SCAN_TO_IMAGE_H
