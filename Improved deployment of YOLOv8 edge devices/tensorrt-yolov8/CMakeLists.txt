cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(yolov8_detect LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: 
${CUDA_LIBRARIES}
")
message(STATUS "CUDA Headers: 
${CUDA_INCLUDE_DIRS}
")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: 
${OpenCV_LIBS}
")
message(STATUS "OpenCV Libraries: 
${OpenCV_LIBRARIES}
")
message(STATUS "OpenCV Headers: 
${OpenCV_INCLUDE_DIRS}
")

# TensorRT
set(TensorRT_INCLUDE_DIRS /usr/src/TensorRT)
set(TensorRT_LIBRARIES /usr/src/TensorRT)

message(STATUS "TensorRT Libs: 
${TensorRT_LIBRARIES}
")
message(STATUS "TensorRT Headers: 
${TensorRT_INCLUDE_DIRS}
")

list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ./include
        )

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        )

include_directories(${INCLUDE_DIRS})

add_library(yolov8_detect SHARED
        yolov8.cpp
        include/yolov8.hpp
        include/common.hpp
        )

link_directories(${ALL_LIBS})
target_link_libraries(yolov8_detect PRIVATE nvinfer nvinfer_plugin ${CUDA_LIBRARIES} ${OpenCV_LIBS})

