cmake_minimum_required(VERSION 3.5)
project(ctdet_trt)
find_package(CUDA REQUIRED)

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include/)

message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
message(STATUS "Found TensorRT libs ${TENSORRT_LIBRARY_INFER}")


# set your OPENCV path
# set(OpenCV_DIR /home/xxx/libs/opencv-3.4.2/build)
find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARIES_DIRS})

file(GLOB CPP_SRC *.cpp)
file(GLOB CU_SRC *.cu)

SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast")
list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")


message(STATUS "cuda include path is <<<<<<<<<<", ${CUDA_INCLUDE_DIRS})


include_directories(${CUDA_INCLUDE_DIRS}
                    ${CUDA_ARCH64_PATH}
                    ${TENSORRT_INCLUDE_DIR}
                    ${OpenCV_INCLUDE_DIRS}
                    ../include
                    ../include/common
                    ../onnx-tensorrt
                    ../plugins/include
                    ../plugins/parsers/common
                    ../plugins/onnx
                    ../plugins/plugin
                    ../plugins/plugin/common)

set(PLUGIN_LISTS
    ../plugins/plugin/DCNv2
    ../plugins/plugin/nmsPlugin
    ../plugins/plugin/normalizePlugin
    ../plugins/plugin/priorBoxPlugin
    ../plugins/plugin/reorgPlugin
    ../plugins/plugin/gridAnchorPlugin
    ../plugins/plugin/regionPlugin
    ../plugins/plugin/nvFasterRCNN
    ../plugins/plugin/batchedNMSPlugin
    ../plugins/plugin/flattenConcat
    ../plugins/plugin/coordConvACPlugin
    ../plugins/plugin/cropAndResizePlugin
    ../plugins/plugin/proposalPlugin
    ../plugins/plugin/efficientNMSPlugin
    ../plugins/plugin/groupNormalizationPlugin
    ../plugins/plugin/leakyReluPlugin
    ../plugins/plugin/splitPlugin
    ../plugins/plugin/scatterPlugin
    ../plugins/plugin/batchTilePlugin
    ../plugins/plugin/detectionLayerPlugin
    ../plugins/plugin/proposalLayerPlugin
    ../plugins/plugin/pyramidROIAlignPlugin
    ../plugins/plugin/generateDetectionPlugin
    ../plugins/plugin/multilevelProposeROI
    ../plugins/plugin/multilevelCropAndResizePlugin
    ../plugins/plugin/resizeNearestPlugin
    ../plugins/plugin/specialSlicePlugin
    ../plugins/plugin/instanceNormalizationPlugin
    )

    include_directories(../plugins/plugin/common 
    ../plugins/plugin/common/kernels 
    ../plugins/samples/common)

#set(ALL_SRCS)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    #message(STATUS "PLUGIN_ITER:${PLUGIN_ITER} \n")
    include_directories(${PLUGIN_ITER})
    #file(GLOB SRC_CU_PLUGINS ${PLUGIN_ITER}/*cu)
    #file(GLOB SRC_CPP_PLUGINS ${PLUGIN_ITER}/*cpp)
    #list(APPEND ALL_SRCS "${PLUGIN_ITER}/*.cu")     
endforeach(PLUGIN_ITER)

file(GLOB ALL_SRCS 
    ../plugins/plugin/DCNv2/*.cpp
    ../plugins/plugin/nmsPlugin/*.cpp
    ../plugins/plugin/normalizePlugin/*.cpp
    ../plugins/plugin/priorBoxPlugin/*.cpp
    ../plugins/plugin/reorgPlugin/*.cpp
    ../plugins/plugin/gridAnchorPlugin/*.cpp
    ../plugins/plugin/regionPlugin/*.cpp
    ../plugins/plugin/nvFasterRCNN/*.cpp
    ../plugins/plugin/batchedNMSPlugin/*.cpp
    ../plugins/plugin/flattenConcat/*.cpp
    ../plugins/plugin/coordConvACPlugin/*.cpp
    ../plugins/plugin/cropAndResizePlugin/*.cpp
    ../plugins/plugin/proposalPlugin/*.cpp
    ../plugins/plugin/efficientNMSPlugin/*.cpp
    ../plugins/plugin/groupNormalizationPlugin/*.cpp
    ../plugins/plugin/leakyReluPlugin/*.cpp
    ../plugins/plugin/splitPlugin/*.cpp
    ../plugins/plugin/scatterPlugin/*.cpp
    ../plugins/plugin/batchTilePlugin/*.cpp
    ../plugins/plugin/detectionLayerPlugin/*.cpp
    ../plugins/plugin/proposalLayerPlugin/*.cpp
    ../plugins/plugin/pyramidROIAlignPlugin/*.cpp
    ../plugins/plugin/generateDetectionPlugin/*.cpp
    ../plugins/plugin/multilevelProposeROI/*.cpp
    ../plugins/plugin/multilevelCropAndResizePlugin/*.cpp
    ../plugins/plugin/resizeNearestPlugin/*.cpp
    ../plugins/plugin/specialSlicePlugin/*.cpp
    ../plugins/plugin/instanceNormalizationPlugin/*.cpp
    ../plugins/plugin/common/kernels/kernel.cpp


    ../plugins/plugin/DCNv2/*.cu
    ../plugins/plugin/nmsPlugin/*.cu
    ../plugins/plugin/normalizePlugin/*.cu
    ../plugins/plugin/priorBoxPlugin/*.cu
    ../plugins/plugin/reorgPlugin/*.cu
    ../plugins/plugin/gridAnchorPlugin/*.cu
    ../plugins/plugin/regionPlugin/*.cu
    ../plugins/plugin/nvFasterRCNN/*.cu
    ../plugins/plugin/batchedNMSPlugin/*.cu
    ../plugins/plugin/flattenConcat/*.cu
    ../plugins/plugin/coordConvACPlugin/*.cu
    ../plugins/plugin/cropAndResizePlugin/*.cu
    ../plugins/plugin/proposalPlugin/*.cu
    ../plugins/plugin/efficientNMSPlugin/*.cu
    ../plugins/plugin/groupNormalizationPlugin/*.cu
    ../plugins/plugin/leakyReluPlugin/*.cu
    ../plugins/plugin/splitPlugin/*.cu
    ../plugins/plugin/scatterPlugin/*.cu
    ../plugins/plugin/batchTilePlugin/*.cu
    ../plugins/plugin/detectionLayerPlugin/*.cu
    ../plugins/plugin/proposalLayerPlugin/*.cu
    ../plugins/plugin/pyramidROIAlignPlugin/*.cu
    ../plugins/plugin/generateDetectionPlugin/*.cu
    ../plugins/plugin/multilevelProposeROI/*.cu
    ../plugins/plugin/multilevelCropAndResizePlugin/*.cu
    ../plugins/plugin/resizeNearestPlugin/*.cu
    ../plugins/plugin/specialSlicePlugin/*.cu
    ../plugins/plugin/instanceNormalizationPlugin/*.cu
    ../plugins/plugin/common/*.cpp
    ../plugins/plugin/common/kernels/*.cu
    )



#message(STATUS "ALL_SRCS:${ALL_SRCS} \n")




message(STATUS "TENSORRT_LIBRARY_INFER:${TENSORRT_LIBRARY_INFER}")
cuda_add_library(ctdet SHARED ${ALL_SRCS} ${CPP_SRC} ${CU_SRC})
target_link_libraries(ctdet
        #PRIVATE CUDA::cudart CUDA::cuda_driver
        ${TENSORRT_LIBRARY_INFER}
        ${OpenCV_LIBS}
        nvonnxparser
        #cublas
        )
