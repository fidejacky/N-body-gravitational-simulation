cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120-real)
endif()

project(nbody LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/results)

add_executable(nbody_cpu src/nbody_cpu.cpp)
target_compile_features(nbody_cpu PRIVATE cxx_std_17)

add_executable(nbody_naive src/nbody_naive.cu)

add_executable(nbody_tiled src/nbody_tiled.cu)

add_executable(nbody_tiled_fastmath src/nbody_tiled_fastmath.cu)

add_executable(nbody_tiled_fastmath_um src/nbody_tiled_fastmath.cu)
target_compile_options(nbody_tiled_fastmath_um PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

add_executable(nbody_tuned src/nbody_tuned.cu)

add_executable(nbody_thrust src/nbody_thrust.cu)
if(MSVC)
  target_compile_options(nbody_thrust PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

add_executable(validate src/validate.cpp)
target_compile_features(validate PRIVATE cxx_std_17)
