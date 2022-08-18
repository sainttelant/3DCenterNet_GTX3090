# TensorRT-CenterNet-3D   
Most parts of this Repo are based on [CaoWGG](https://github.com/CaoWGG/TensorRT-CenterNet), Thanks for that great work.

Attention:::



mv /usr/lib/libcuda.so /usr/lib/libcuda.so.bak 
then run program!

### 1. Enviroments
```
ubuntu 18.04
TensorRT 8.0.1.6
jetson nx 4.6
onnx-tensorrt for tensorrt8,please refer to other repos, onnx-tensorrt repo
cuda 10.2.30
opencv4.1.1

pip show torch
Name: torch
Version: 1.7.0
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Location: /home/nvidia/.local/lib/python3.6/site-packages
Requires: dataclasses, future, numpy, typing-extensions
Required-by: torchvision

Version: 1.19.4
Summary: NumPy is the fundamental package for array computing with Python.
Home-page: https://www.numpy.org
Author: Travis E. Oliphant et al.
Author-email: 
License: BSD
Location: /usr/local/lib/python3.6/dist-packages
Requires: 
Required-by: onnx, torch, torchvision, uff

Name: onnx
Version: 1.9.0
Summary: Open Neural Network Exchange
Home-page: https://github.com/onnx/onnx
Author: ONNX
Author-email: onnx-technical-discuss@lists.lfai.foundation
License: Apache License v2.0
Location: /usr/local/lib/python3.6/dist-packages
Requires: numpy, protobuf, six, typing-extensions
Required-by: 



nxjetson version 
-- ******** Summary ********
--   CMake version         : 3.24.0
--   CMake command         : /usr/local/bin/cmake
--   System                : Linux
--   C++ compiler          : /usr/bin/c++
--   C++ compiler version  : 7.5.0
--   CXX flags             :  -Wall -Wno-deprecated-declarations -Wno-unused-function -Wnon-virtual-dtor
--   Build type            : Debug
--   Compile definitions   : ONNX_NAMESPACE=onnx2trt_onnx
--   CMAKE_PREFIX_PATH     : 
--   CMAKE_INSTALL_PREFIX  : /home/nvidia/xuewei/3D/build/..
--   CMAKE_MODULE_PATH     : 
-- 
--   ONNX version          : 1.8.0
--   ONNX NAMESPACE        : onnx2trt_onnx
--   ONNX_BUILD_TESTS      : OFF
--   ONNX_BUILD_BENCHMARKS : OFF
--   ONNX_USE_LITE_PROTO   : OFF
--   ONNXIFI_DUMMY_BACKEND : OFF
--   ONNXIFI_ENABLE_EXT    : OFF
-- 
--   Protobuf compiler     : /usr/local/bin/protoc
--   Protobuf includes     : /usr/local/include
--   Protobuf libraries    : /usr/local/lib/libprotobuf.so;-lpthread
--   BUILD_ONNX_PYTHON     : OFF
-- Found TensorRT headers at /usr/include/aarch64-linux-gnu
-- Find TensorRT libs at /usr/lib/aarch64-linux-gnu/libnvinfer.so;/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so
-- Found CUDA: /usr/local/cuda-10.2 (found version "10.2") 
-- Found TensorRT headers at /usr/include/aarch64-linux-gnu
-- Found TensorRT libs /usr/lib/aarch64-linux-gnu/libnvinfer.so
-- TENSORRT_LIBRARY_INFER:/usr/lib/aarch64-linux-gnu/libnvinfer.so
CMake Warning (dev) at example/CMakeLists.txt:12:
  Syntax Warning in cmake code at column 49

  Argument not separated from preceding token by whitespace.
This warning is for project developers.  Use -Wno-dev to suppress it.

-- cuda aarch64 path is <<<<<<<<<<,/usr/local/cuda-10.2/targets/aarch64-linux/include
-- INCLUDE FIND_LIBRARY_CREATE=============

-- INCLUDE FIND_LIBRARY_CREATE=============

-- CMAKE_BINARY_DIR/home/nvidia/xuewei/3D/build
Building for TensorRT version: 8.2.1, library version: 8
-- 
-- Targeting TRT Platform: aarch64
Building in debug mode 
-- CUDA version set to 10.2
-- 
-- cuDNN version set to 8.2
-- 
-- Protobuf version set to 3.15.8
-- set cub root dir/home/nvidia/xuewei/3D/plugins/plugin/cub
-- Found CUDA: /usr/local/cuda-10.2 (found suitable version "10.2", minimum required is "10.2") 
-- PARSE_INCLUDE===========/home/nvidia/xuewei/3D/plugins/parsers/common
-- PATH OF TRT_LIB_DIR/home/nvidia/xuewei/3D/build
-- ========================= Importing and creating target nvinfer ==========================
-- Looking for library nvinfer
-- Library that was found /usr/lib/aarch64-linux-gnu/libnvinfer.so
-- ==========================================================================================
-- ========================= Importing and creating target nvuffparser ==========================
-- Looking for library nvparsers
-- Library that was found /usr/lib/aarch64-linux-gnu/libnvparsers.so
-- ==========================================================================================
-- GPU_ARCHS defined as 72. Generating CUDA code for SM 72
--  CUB_PATH IS:<<<<<<<<<<<<wilson:/home/nvidia/xuewei/3D/plugins/plugin/cub
--  CUB_PATH IS:/home/nvidia/xuewei/3D/plugins/plugin/cub
-- BERT_CU_SOURCES IS DISABLE<<<<<<<<<<<<<<<<<<<<<<<<<< 

-- ========================= Importing and creating target nvcaffeparser ==========================
-- Looking for library nvparsers
-- Library that was found /usr/lib/aarch64-linux-gnu/libnvparsers.so
-- ==========================================================================================
-- Configuring done


### 2. Performance



### 3. How to use
- download the `ddd_3dop.pth` from [here](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md)
- Convert CenterNet-3D model to `onnx`. See [here](readme/ctddd2onnx.md) for details, Use [netron](https://github.com/lutzroeder/netron) to observe whether the output of the converted onnx model is `(('hm', 3), ('dep', 1), ('rot', 8), ('dim', 3), ('wh', 2), ('reg', 2))`  
- set the OPENCV path in `./src/CMakeLists.txt` ,set the TENSORRT path in `./CMakeLists.txt`
- uncomment the configs of 3-D model in `./include/ctdetConfig.h` and comment others.
- build
```
git clone https://github.com/Qjizhi/TensorRT-CenterNet-3D.git
cd TensorRT-CenterNet-3D
mkdir build
cd build && cmake .. && make
cd ..
```
- build the engie and do the  inference
```
./buildEngine -i model/ddd_3dop.onnx -o model/ddd_3dop.engine
./runDet -e model/ddd_3dop.engine -i 000292.png -c test.h264
```


#### 4. Good to know
- [x] This codes are just for testing and will not be well maintained, for any suggestions and improvements, please propose an issue.
- [x] With the help of TensorRT, the inference time has been shortened from **0.227s** to **98.89ms**.
- [x] The functions of previous Repo are still remained, for exmaple, centerface...




### Related projects
* [TensorRT-CenterNet](https://github.com/CaoWGG/TensorRT-CenterNet)
* [TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3)
* [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)
* [TensorRT](https://github.com/NVIDIA/TensorRT)
* [CenterNet](https://github.com/xingyizhou/centernet)
* [centerface](https://github.com/Star-Clouds/centerface)
* [netron](https://github.com/lutzroeder/netron)
* [cpp-optparse](https://github.com/weisslj/cpp-optparse)
