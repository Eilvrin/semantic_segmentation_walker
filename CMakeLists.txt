cmake_minimum_required(VERSION 2.8.3)
project(semantic_segmentation)

# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  cmake_modules
  sensor_msgs
  pcl_ros
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)

add_definitions(${Caffe_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Enable optimization for eigen
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -msse2 -march=native")
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -msse2 -march=native")

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp tf sensor_msgs pcl_ros
  DEPENDS Caffe OpenCV
)

include_directories(include ${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS})

add_executable(semantic_seg_node src/semantic_seg_node.cpp src/scan_to_image.cpp src/CaffeWrapper.cpp)
target_link_libraries(semantic_seg_node ${catkin_LIBRARIES} ${Eigen_LIBRARIES} ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})

#Installation 
install(TARGETS semantic_seg_node 
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
