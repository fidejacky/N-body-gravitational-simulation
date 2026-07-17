cmake_minimum_required(VERSION 3.24)

# "native" auto-detection has been unreliable in this environment (it resolved
# to sm_75 on an RTX 5050, a Blackwell/sm_120 GPU), forcing a PTX JIT path that
# the installed driver's toolchain couldn't parse. Pin the real architecture
# explicitly; "-real" embeds only native SASS, skipping PTX JIT entirely. This
# must be set before project() enables CUDA, since project() auto-detects and
# caches CMAKE_CUDA_ARCHITECTURES itself the moment it identifies the compiler,
# before any if(NOT CMAKE_CUDA_ARCHITECTURES) check placed after it would run.
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

# Stage 0: CPU baseline (no CUDA)
add_executable(nbody_cpu src/nbody_cpu.cpp)
target_compile_features(nbody_cpu PRIVATE cxx_std_17)

# Stage 1: naive GPU implementation
add_executable(nbody_naive src/nbody_naive.cu)

# Stage 2: shared-memory tiling
add_executable(nbody_tiled src/nbody_tiled.cu)

# Stage 3: loop unrolling + rsqrtf/fmaf intrinsics
add_executable(nbody_tiled_fastmath src/nbody_tiled_fastmath.cu)

# Same Stage 3 source, additionally compiled with --use_fast_math, to measure
# the accuracy/speed tradeoff of the compiler's aggressive math flag on top of
# the manual intrinsics, as called out in project_plan.md.
add_executable(nbody_tiled_fastmath_um src/nbody_tiled_fastmath.cu)
target_compile_options(nbody_tiled_fastmath_um PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# Stage 4: block-size and occupancy tuning. Same Stage 3 kernel, templated on
# block size so one binary can sweep 64/128/256/512/1024 via --block-size
# while the compiler still fully unrolls the inner tile loop per size.
add_executable(nbody_tuned src/nbody_tuned.cu)

# Stage 5: library-based performance ceiling via Thrust (validation/ceiling
# use only, per CLAUDE.md — not the main implementation). Thrust's headers
# require the standard-conforming preprocessor; MSVC defaults to its
# traditional one, so this must be passed through to cl.exe for this target.
add_executable(nbody_thrust src/nbody_thrust.cu)
if(MSVC)
  target_compile_options(nbody_thrust PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

# Cross-stage correctness check (no CUDA needed): compares two --dump-final CSVs
add_executable(validate src/validate.cpp)
target_compile_features(validate PRIVATE cxx_std_17)
