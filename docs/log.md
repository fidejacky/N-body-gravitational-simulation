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
- Next step: shared-memory tiling (Stage 2).

## Stage 1: correctness validated against the CPU baseline
- Added `--dump-final <path>` to both `nbody_cpu.exe` and `nbody_naive.exe`, writing final body positions to CSV (shared `writePositionsCsv`/`readPositionsCsv` helpers in `common.h`). Both stages share the same seeded RNG for initial conditions, so body `i` in one dump corresponds directly to body `i` in the other.
- Added a standalone `validate` tool (`src/validate.cpp`, pure host C++, no CUDA) that compares two dumps body-by-body and reports max/mean positional deviation against a tolerance.
- Important gotcha: `nbody_naive.exe`'s warmup loop runs real physics steps on the same buffers before the timed section (it is not a throwaway measurement warmup), so `--warmup 0` must be passed when validating, otherwise the GPU run advances more steps than the CPU run and the comparison is meaningless.
- At N=1024/4096, short runs (10-30 steps) agree within float32 rounding noise: max positional deviation ~0.00015 at 10 steps, ~0.0033 at 30 steps, well under a `0.01` tolerance. This confirms the naive kernel's force calculation and integration are implemented correctly.
- At longer runs, deviation grows roughly exponentially (N=4096: `0.12` at 50 steps, `1.34` at 70 steps, `2.43` at 100 steps) and fails the tolerance. This is expected chaotic divergence, not a bug: gravitational N-body systems are chaotically sensitive to tiny numerical differences (here, float32 GPU accumulation vs. double-precision CPU accumulation), especially around close encounters, and no two floating-point implementations of a chaotic system stay in agreement indefinitely. Documented in the README so this isn't mistaken for a regression later.
- Next step: shared-memory tiling (Stage 2).
