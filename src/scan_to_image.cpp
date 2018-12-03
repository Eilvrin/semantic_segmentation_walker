#include "semantic_seg/scan_to_image.h"

namespace semantic_seg {

ScanToImage::ScanToImage(ros::NodeHandle node, ros::NodeHandle private_nh)
{
  private_nh.param("target_frame_id", target_frame_id_, std::string("velodyne_static"));
  private_nh.param("width", width_, 450);
  private_nh.param("height", height_, 252);

  // Parameters for spherical projection
  private_nh.param("x_angle_range", xAngleRange_, static_cast<float>(2.0*M_PI));
  xAngularResolution_ = xAngleRange_ / width_;
  private_nh.param("y_angular_resolution", yAngularResolution_, 0.02321288f);
  private_nh.param("x_angle_offset", xAngleOffset_, 0.0f);
  private_nh.param("y_angle_offset", yAngleOffset_, 4.45059f);


  private_nh.param("model_prototxt", model_prototxt_, std::string());
  private_nh.param("weights_caffemodel", weights_caffemodel_, std::string());
  private_nh.param("use_gpu", use_gpu_, false);
  private_nh.param("num_classes", num_classes_, 0);
  caffe_wrapper.Initialize(model_prototxt_, weights_caffemodel_, use_gpu_);

  // Resize vectors
  depth_.resize(width_*height_, 0);
  idx_.resize(width_*height_, 0);
  normals_xyz_.resize(width_*height_*3, 0);

  assembled_cloud_ =
      node.subscribe("assembled_cloud", 10, &ScanToImage::processScan, this);
  labeled_cloud_pub_ =
      node.advertise<sensor_msgs::PointCloud2>("labeled_point_clouds", 10);
  unlabeled_cloud_pub_ =
      node.advertise<sensor_msgs::PointCloud2>("unlabeled_point_clouds", 10);
}

void ScanToImage::processScan(const PclCloudPtr &point_cloud)
{
  if(point_cloud->points.size() == 0) {
    return;
  }
  auto totalstart = std::chrono::system_clock::now();
  auto wcts = std::chrono::system_clock::now();
  
  PclCloudPtr transformed_cloud (new PclCloud);
  std::fill(depth_.begin(), depth_.end(), std::numeric_limits<float>::quiet_NaN());
  normals_xyz_.resize(width_*height_*3, 0);
  std::fill(normals_xyz_.begin(), normals_xyz_.end(),  std::numeric_limits<float>::quiet_NaN());
  std::fill(idx_.begin(), idx_.end(),  std::numeric_limits<float>::quiet_NaN());

  std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
  std::cout << "Initialization finished in " << wctduration.count() << " seconds" << std::endl;


  wcts = std::chrono::system_clock::now();

  if (target_frame_id_ == "") target_frame_id_ = point_cloud->header.frame_id;
  if(!pcl_ros::transformPointCloudWithNormals (target_frame_id_, *point_cloud, *transformed_cloud, listener_)){
    ROS_WARN("Transform for PointCloud is not available.");
    return;
  }

  for(int i=0; i<transformed_cloud->size(); ++i){
    // Reset intentity to Unlabeled class
    transformed_cloud->points[i].intensity = num_classes_;
    
    // Skip nan points
    if (std::isnan(transformed_cloud->points[i].x) || std::isnan(transformed_cloud->points[i].y) || std::isnan(transformed_cloud->points[i].z)) continue;


    Eigen::Vector2i projected;
    projected = getProjectedIndexVelodyne(transformed_cloud->points[i].getVector3fMap());
  
    // Check boundary conditions
    if( projected(1) < 0 || projected(1) >= height_ ) continue;
    if( projected(0) < 0 || projected(0) >= width_  ) continue;

    float distance = transformed_cloud->points[i].getVector3fMap().norm();

    uint32_t index;
    index = (width_ - 1 - projected(0))+(height_ - 1 - projected(1))*width_;

    if (std::isnan(depth_[index]) || depth_[index] > distance ){
      // Index image
      idx_[index] = i;
      // Depth image
      depth_[index] = distance;
      // Normals image
      normals_xyz_[index+0*(width_*height_)] = transformed_cloud->points[i].normal_z; // B
      normals_xyz_[index+1*(width_*height_)] = transformed_cloud->points[i].normal_y; // G
      normals_xyz_[index+2*(width_*height_)] = transformed_cloud->points[i].normal_x; // R
    }
  }
  
  wctduration = (std::chrono::system_clock::now() - wcts);
  std::cout << "Projection finished in " << wctduration.count() << " seconds" << std::endl;

  wcts = std::chrono::system_clock::now();
  // If value in depth is more than 100
  // set it to 100
  for(auto &d : depth_) {
    d = d > MAX_RANGE_ ? MAX_RANGE_ : d;
    d = (MAX_RANGE_-d)/MAX_RANGE_;
    d = std::isnan(d) ? 0 : d;
  }
  for(auto &n : normals_xyz_){
    n = std::isnan(n) ? 0 : n;
  }

  normals_xyz_.insert(normals_xyz_.end(), depth_.begin(), depth_.end());

  wctduration = (std::chrono::system_clock::now() - wcts);
  std::cout << "Normalization finished in " << wctduration.count() << " seconds" << std::endl;

  wcts = std::chrono::system_clock::now();
  auto out = caffe_wrapper.Segment(normals_xyz_);
  wctduration = (std::chrono::system_clock::now() - wcts);
  std::cout << "Forward pass finished in " << wctduration.count() << " seconds" << std::endl;

  PclCloudPtr labeled_cloud (new PclCloud);
  PclCloudPtr unlabeled_cloud (new PclCloud);

  if(out.size() != idx_.size())
    throw std::runtime_error("Output of neural network and index image are different sizes.");

  for(int i=0; i < out.size(); ++i){
    float& index = idx_[i];
    float& label = out[i];
    if(std::isnan(index)) continue;
    auto& p = transformed_cloud->points[index];
    p.intensity = label;
    labeled_cloud->points.emplace_back(p);
  }

  for(const auto& p : transformed_cloud->points){
    if(p.intensity == num_classes_){
      unlabeled_cloud->points.emplace_back(p);
    }
  }

  labeled_cloud->header.frame_id = transformed_cloud->header.frame_id;
  labeled_cloud->header.stamp = transformed_cloud->header.stamp;
  labeled_cloud->header.seq = transformed_cloud->header.seq;

  unlabeled_cloud->header.frame_id = transformed_cloud->header.frame_id;
  unlabeled_cloud->header.stamp = transformed_cloud->header.stamp;
  unlabeled_cloud->header.seq = transformed_cloud->header.seq;

  labeled_cloud_pub_.publish(*labeled_cloud);
  unlabeled_cloud_pub_.publish(*unlabeled_cloud);
  
  std::chrono::duration<double> totalduration = (std::chrono::system_clock::now() - totalstart);
  std::cout << "Total time " << totalduration.count() << " seconds" << std::endl;

}

Eigen::Vector2i ScanToImage::getProjectedIndexVelodyne(const Eigen::Vector3f& point){
  Eigen::Vector2i index;
  float angleX = atan2(point.y(), point.x());
  float angleY = asin(point.z() / point.norm());

  index(1) = std::floor(normalizeAngle(angleY - yAngleOffset_) / yAngularResolution_);
  index(0) = std::floor(normalizeAngle(angleX + xAngleOffset_) / xAngularResolution_);
  return index;
}

float ScanToImage::normalizeAngle(float angle)
{
  static float TwoPi = 2.0*M_PI;
  angle = std::fmod(angle, TwoPi);
  if(angle < 0.0){
    angle += TwoPi;
  }
  return angle;
}

} // namespace semantic_seg
