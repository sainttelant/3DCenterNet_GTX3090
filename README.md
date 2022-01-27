# TensorRT-CenterNet-3D   
Most parts of this Repo are based on [CaoWGG](https://github.com/CaoWGG/TensorRT-CenterNet), Thanks for that great work.
### 1. Enviroments
```
ubuntu 18.04
TensorRT 8.2GA
onnx-tensorrt for tensorrt8,please refer to other repos, onnx-tensorrt repo
cuda 11.2

pip show torch
Name: torch
Version: 1.7.0+cu110
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Location: /home/{Path}/anaconda3/envs/CenterTrack/lib/python3.6/site-packages
Requires: numpy, typing-extensions, future, dataclasses
Required-by: torchvision, torchaudio

GTX3090

-- The C compiler identification is GNU 7.5.0
-- The CXX compiler identification is GNU 7.5.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Protobuf: /usr/local/lib/libprotobuf.so;-lpthread (found version "3.15.8") 
Generated: /home/{Path}/TensorRT-CenterNet-3D-master/build/onnx-tensorrt/third_party/onnx/onnx/onnx_onnx2trt_onnx-ml.proto
Generated: /home/{Path}/TensorRT-CenterNet-3D-master/build/onnx-tensorrt/third_party/onnx/onnx/onnx-operators_onnx2trt_onnx-ml.proto
Generated: /home/{Path}/TensorRT-CenterNet-3D-master/build/onnx-tensorrt/third_party/onnx/onnx/onnx-data_onnx2trt_onnx.proto
-- 
-- ******** Summary ********
--   CMake version         : 3.22.2
--   CMake command         : /snap/cmake/1005/bin/cmake
--   System                : Linux
--   C++ compiler          : /usr/bin/c++
--   C++ compiler version  : 7.5.0
--   CXX flags             :  -Wall -Wno-deprecated-declarations -Wno-unused-function -Wnon-virtual-dtor
--   Build type            : Release
--   Compile definitions   : SOURCE_LENGTH=42;ONNX_NAMESPACE=onnx2trt_onnx
--   CMAKE_PREFIX_PATH     : 
--   CMAKE_INSTALL_PREFIX  : /usr/local
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
-- Found CUDA headers at /usr/local/cuda/include
-- Found TensorRT headers at /usr/include/x86_64-linux-gnu
-- Find TensorRT libs at /usr/lib/x86_64-linux-gnu/libnvinfer.so;/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
-- Found TENSORRT: /usr/include/x86_64-linux-gnu  
-- Found Threads: TRUE  
-- Found CUDA: /usr/local/cuda-11.2 (found version "11.2") 
-- Found TensorRT headers at /usr/include/x86_64-linux-gnu
-- Found TensorRT libs /usr/lib/x86_64-linux-gnu/libnvinfer.so
-- Found OpenCV: /usr/local (found version "4.5.1") 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/{Path}/TensorRT-CenterNet-3D-master/build
```

``` 
    And i use CenterTrack virtual environments to compile the programs, py3.6 version
```


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
