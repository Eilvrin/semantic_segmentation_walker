#include <ros/ros.h>
#include "semantic_seg/scan_to_image.h"

int main (int argc, char **argv)
{
  ros::init(argc,argv,"semantic_seg");
  ros::NodeHandle n;
  ros::NodeHandle private_nh("~");

  semantic_seg::ScanToImage scan_to_img (n, private_nh);
  
  // handle callbacks until shut down
  ros::spin();
  return 0;
}
