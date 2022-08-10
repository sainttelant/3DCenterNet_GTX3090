cmake_minimum_required(VERSION 3.5)
project(example)


SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(CUDA_ARCH64_PATH /usr/local/cuda-10.2/targets/aarch64-linux/include)
message(STATUS "cuda aarch64 path is <<<<<<<<<<", ${CUDA_ARCH64_PATH})

include_directories(../include
                    ${CUDA_ARCH64_PATH}
                    ../onnx-tensorrt)

add_executable(buildEngine buildEngine.cpp)
target_link_libraries(buildEngine ctdet)

add_executable(runDet runDet.cpp)
target_link_libraries(runDet ctdet)
