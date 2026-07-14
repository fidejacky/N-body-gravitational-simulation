cmake_minimum_required(VERSION 3.24)
project(nbody LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/results)

# Stage 0: CPU baseline (no CUDA)
add_executable(nbody_cpu src/nbody_cpu.cpp)
target_compile_features(nbody_cpu PRIVATE cxx_std_17)

# Stage 1: naive GPU implementation
add_executable(nbody_naive src/nbody_naive.cu)
set_target_properties(nbody_naive PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
