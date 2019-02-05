# Semantic segmentation

### Overview
This ROS package contains the implementation of MultiNet which can be run directly on the walker.

![youtubevideo](https://github.com/Eilvrin/semantic_segmentation_walker/blob/master/semantic_segmentation_point_clouds.png)
[See on youtube](https://www.youtube.com/embed/2lej5Ka1Rvw)

### Caffe installation

Prerequisites:
```
    sudo apt-get install protobuf-compiler libprotobuf-dev libgflags-dev libatlas-dev libatlas-base-dev libopenblas-dev liblmdb-dev libleveldb-dev libsnappy-dev libgoogle-glog-dev libopencv-dev libhdf5-serial-dev libboost-all-dev
```

Either clone the caffe repository from https://github.com/BVLC/caffe.git or extract `nets/caffe_multinet.7z` into an external folder. Compile the Caffe:
```
    cd caffe
    mkdir build
    cd build
    cmake-gui ..
    make install
    make runtest
```
In the cmake GUI, set the build type to `Release` and enable `OpenMP`. For the GPU version turn `CPU_ONLY` `OFF`, `USE_CUDNN` `ON`.

If cudnn is not installed, install it as described [here] (https://stackoverflow.com/questions/31326015/how-to-verify-cudnn-installation).  For `caffe_multinet` the 5.1 version of cudnn is suitable.

For CPU only version turn `CPU_ONLY` `ON`, `USE_CUDNN` `OFF`. 

More instructions on installation of Caffe can be found [here] (http://caffe.berkeleyvision.org/installation.html).

#### Troubleshooting: fatal error: caffe/proto/caffe.pb.h: No such file or directory

As a workaround, create a symlink to the missing headers.
```
cd <...>/caffe/include/caffe
ln -s ../../build/include/caffe/proto .
```



### Launching
For GPU version in the `\semantic_seg\config\scan_to_image.yaml` specify the following parameters:
* `model_prototxt: "nets/deploy.prototxt"`
* `use_gpu: true`

For CPU version the the parameters should be:
* `model_prototxt: "nets/deploy_cpu.prototxt"`
* `use_gpu: false`

To launch only the node with network:
```
    roslaunch semantic_seg semantic_seg.launch
``` 
To launch the node with network and visualization in rviz:
```
    roslaunch semantic_seg semantic_seg_rviz.launch
```
