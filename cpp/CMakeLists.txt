cmake_minimum_required(VERSION 3.20)
project(rclane_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_path(TENSORRT_INCLUDE_DIR NvInferRuntime.h
    HINTS "$ENV{TENSORRT_ROOT}/include"
    PATHS /usr/include /usr/include/aarch64-linux-gnu /usr/local/TensorRT/include
)
find_library(TENSORRT_LIBRARY NAMES nvinfer libnvinfer.so.10
    HINTS "$ENV{TENSORRT_ROOT}/lib" "$ENV{TENSORRT_ROOT}/lib64"
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
    HINTS "$ENV{CUDA_HOME}/include" "$ENV{CUDA_PATH}/include"
    PATHS /usr/local/cuda/include /usr/include
)
find_library(CUDART_LIBRARY NAMES cudart libcudart.so.13 libcudart.so.12
    HINTS "$ENV{CUDA_HOME}/lib64" "$ENV{CUDA_PATH}/lib64"
    PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
          /usr/lib/aarch64-linux-gnu
)

foreach(required_path
        TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY CUDA_INCLUDE_DIR CUDART_LIBRARY)
    if(NOT ${required_path})
        message(FATAL_ERROR
            "${required_path} was not found; pass -D${required_path}=..."
        )
    endif()
endforeach()

find_package(OpenMP REQUIRED)

add_executable(rclane_runtime
    src/bev.cpp
    src/decoder.cpp
    src/main.cpp
    src/preprocess.cpp
    src/tensorrt_runner.cpp
)
target_include_directories(rclane_runtime PRIVATE
    include
    "${TENSORRT_INCLUDE_DIR}"
    "${CUDA_INCLUDE_DIR}"
)
target_link_libraries(rclane_runtime PRIVATE
    "${TENSORRT_LIBRARY}"
    "${CUDART_LIBRARY}"
    OpenMP::OpenMP_CXX
    dl
)
target_compile_options(rclane_runtime PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic
)

get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
get_filename_component(CUDART_LIBRARY_DIR "${CUDART_LIBRARY}" DIRECTORY)
set_target_properties(rclane_runtime PROPERTIES
    BUILD_RPATH "${TENSORRT_LIBRARY_DIR};${CUDART_LIBRARY_DIR}"
)
