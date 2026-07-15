# Project log

## Stage 0: CPU baseline scaffolded and verified
- Created the shared utility header with float4 helpers, initialization routines, timing helpers, and CSV logging support.
- Added a standalone Stage 0 CPU implementation that runs the direct all-pairs force computation, updates velocities and positions with semi-implicit Euler, and reports average milliseconds per step over repeated runs.
- Added a placeholder Stage 1 CUDA source file so the project can continue into the first GPU implementation cleanly.
- Verified the build locally with MSVC from Visual Studio Build Tools and ran the executable successfully.
- Verified benchmark output for a sample run:
  - n=1024
  - avg_ms_per_step=3.2363
  - avg_total_ms=16.1815
- The CSV output was written to results/stage0_sample.csv.
- Next step: implement the Stage 1 naive CUDA kernel and compare its timing against this CPU baseline.

## Stage 1: naive GPU kernel implemented and verified
- Implemented `computeForcesKernel` (one thread per body, output-centric, no atomics, self-interaction cancels naturally via the softened denominator so no `i==j` branch is needed) and `integrateKernel` (semi-implicit Euler) in `src/nbody_naive.cu`.
- CLI and CSV format mirror Stage 0 (`--n`, `--steps`, `--repeats`, `--csv`, `--colliding`, plus `--warmup`) so the two stages are directly comparable.
- Fixed a real bug surfaced by the first actual compile of this file: `common.h`'s custom `float4` struct collided with CUDA's built-in `float4` once `nvcc` compiled real device code. Guarded the custom struct/`make_float4` behind `#ifndef __CUDACC__`.
- Hit and resolved a chain of build/runtime issues, worth remembering for later stages:
  - `CMAKE_CUDA_ARCHITECTURES native` silently resolved to `sm_75` on this machine's actual GPU (RTX 5050 Laptop, compute capability 12.0/Blackwell), confirmed via a `cudaGetDeviceProperties` query. Auto-detection is not trustworthy in this environment; the architecture must be pinned explicitly (`120-real` in CMake, `-arch=sm_120` for raw `nvcc`).
  - Pinning it in `CMakeLists.txt` after `project(... CUDA)` did not work: `project()` auto-detects and caches `CMAKE_CUDA_ARCHITECTURES` itself the moment it identifies the CUDA compiler, before any later `if(NOT CMAKE_CUDA_ARCHITECTURES)` override runs. The `set()` must come before `project()`.
  - CUDA event-based timing (`cudaEventRecord`/`cudaEventSynchronize`) failed at runtime with "the provided PTX was compiled with an unsupported toolchain," even after the architecture fix, while plain kernel launches plus `cudaDeviceSynchronize()` worked correctly. Switched to host-clock timing (`nowMs()`, matching Stage 0) around an explicit synchronize instead of CUDA events, which avoided the issue.
- Verified: `results\nbody_naive.exe --n 8192 --steps 50 --repeats 3` runs successfully at `avg_ms_per_step≈1.39`, versus the CPU baseline's `≈188` at the same N — roughly 135x, a sane result for a first, unoptimized parallelization.
- Not yet done: numerical correctness validation against the CPU baseline (positions within float/double tolerance), per the project's measurement methodology. Good next step before moving to Stage 2.
- Next step: shared-memory tiling (Stage 2).
