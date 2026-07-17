# Direct N-Body Simulation in CUDA

This repository contains a staged GPU Computing project for a direct, all-pairs gravitational N-body simulation written in CUDA C++.

## Project goals
- Implement the simulation in multiple stages, each as a separate compilable version.
- Keep the main computation hand-written in CUDA rather than relying on libraries.
- Benchmark each stage independently and compare it with the CPU baseline.

## Constraints
- Core force kernels are hand-written CUDA C++.
- Thrust/cuBLAS may only be used for validation or performance-ceiling comparisons.
- The GPU kernel uses output-centric parallelization: one thread owns one body.
- A softening term is included via $\epsilon^2$ so close encounters remain finite.
- The simulation uses a SoA-style layout with position and mass packed into a float4.
- The integration scheme is semi-implicit Euler with $G = 1.0$.
- Simulation state stays on the GPU across timesteps unless snapshots are requested.

## Current repository contents
- [src/common.h](src/common.h) - shared helpers for float4 operations, initialization, timing, and CSV output.
- [src/nbody_cpu.cpp](src/nbody_cpu.cpp) - Stage 0 CPU baseline.
- [src/nbody_naive.cu](src/nbody_naive.cu) - Stage 1 naive GPU kernel (one thread per body).
- [src/nbody_tiled.cu](src/nbody_tiled.cu) - Stage 2 shared-memory tiled GPU kernel.
- [src/nbody_tiled_fastmath.cu](src/nbody_tiled_fastmath.cu) - Stage 3 kernel (loop unrolling, `rsqrtf`, `fmaf`); built as two targets, with and without `--use_fast_math`.
- [src/nbody_tuned.cu](src/nbody_tuned.cu) - Stage 4 kernel, templated on block size for a `--block-size` sweep (64/128/256/512/1024).
- [bench/run_sweep.sh](bench/run_sweep.sh) - sweeps `nbody_tuned` across all block sizes and the full N range into one CSV.
- [src/nbody_thrust.cu](src/nbody_thrust.cu) - Stage 5 library-based ceiling, expressed with Thrust instead of a hand-written kernel.
- [src/validate.cpp](src/validate.cpp) - compares two stages' `--dump-final` position dumps against a tolerance.
- [docs/log.md](docs/log.md) - iterative development log.
- [project_plan.md](project_plan.md) - project plan and stage breakdown.
- [CLAUDE.md](CLAUDE.md) - course-project context and current constraints.

## Build

On Windows, both options below need `cl.exe` reachable on PATH first. Open a **"x64 Native Tools Command Prompt for VS"** (or "Developer PowerShell for VS") from the Start Menu; it sets this up automatically for that terminal session. Everything below assumes that terminal.

### Option 1: compile directly with nvcc

Useful for understanding what the toolchain is actually doing, since this is the real compiler command with nothing hidden behind a build system.

```cmd
nvcc -O2 -std=c++17 src\nbody_cpu.cpp -o results\nbody_cpu.exe
nvcc -O2 -std=c++17 -arch=sm_120 src\nbody_naive.cu -o results\nbody_naive.exe
```

- `nvcc` is NVIDIA's CUDA compiler driver. For `nbody_cpu.cpp` (no CUDA code), it just forwards everything to the host compiler (`cl.exe`). For `nbody_naive.cu`, it splits the file: device kernels go to NVIDIA's own backend, host-side code still goes to `cl.exe`.
- `-arch=sm_120` pins the exact compute capability of the GPU running the build (an RTX 5050, Blackwell, compute capability 12.0). Query your own GPU's capability with a tiny `cudaGetDeviceProperties` call if it differs, and swap the number accordingly. `-arch=native`, which is supposed to auto-detect this, was tried first but proved unreliable in this environment: it silently resolved to `sm_75` (a much older architecture) both via plain `nvcc` and via CMake's `CMAKE_CUDA_ARCHITECTURES native`, which forced a PTX JIT recompile at runtime that the installed driver's toolchain couldn't parse, so pin the number explicitly instead of relying on auto-detection.
- Output executables land in `results\`.

### Option 2: one-command build via CMake + Ninja

Prerequisites (one-time per machine): [CMake](https://cmake.org/) 3.24+, [Ninja](https://ninja-build.org/), a CUDA toolkit with `nvcc`, and MSVC Build Tools on Windows.

```powershell
# Windows
.\build.ps1
```

```bash
# Linux/macOS
./build.sh
```

Both scripts configure and build every stage (`nbody_cpu`, `nbody_naive`, ...) in one command via CMake + Ninja, writing binaries to `results/`. Under the hood, this runs the same `nvcc`/`cl` invocations as Option 1, generated automatically from [CMakeLists.txt](CMakeLists.txt) instead of typed by hand. On Windows, `build.ps1` locates and loads the MSVC developer environment itself via `vswhere`, so unlike Option 1, it does not require opening a Developer Command Prompt manually.

### Running a stage

**Stage 0 (CPU baseline)** runs a real simulation and prints timing:

```powershell
results\nbody_cpu.exe --n 1024 --steps 5 --repeats 1 --csv results\stage0_sample.csv
```

Expected output:

```text
n=1024 avg_ms_per_step=... avg_total_ms=...
```

The same row also gets appended to the CSV path passed via `--csv` (defaults to `results\stage0_benchmark.csv` if omitted).

**Stage 1 (naive GPU kernel)** runs the same simulation on the GPU, one thread per body:

```powershell
results\nbody_naive.exe --n 8192 --steps 50 --repeats 3 --csv results\stage1_sample.csv
```

Expected output:

```text
n=8192 avg_ms_per_step=... avg_total_ms=...
```

Same CSV format as Stage 0, appended to the path passed via `--csv` (defaults to `results\stage1_benchmark.csv`), so the two stages can be compared directly.

**Stage 2 (shared-memory tiling)** loads each tile of bodies into `__shared__` memory once per block instead of once per thread, cutting global memory traffic:

```powershell
results\nbody_tiled.exe --n 8192 --steps 50 --repeats 3 --csv results\stage2_sample.csv
```

Same CLI/CSV format as Stages 0 and 1, defaulting to `results\stage2_benchmark.csv`.

**Stage 3 (loop unrolling + fast-math intrinsics)** adds `#pragma unroll` on the inner tile loop and replaces `1.0f / sqrtf(...)` with the `rsqrtf` intrinsic plus `fmaf` for the accumulation, on top of Stage 2's tiling. Built as two executables from the same source so the compiler flag's effect can be measured separately from the manual changes:

```powershell
results\nbody_tiled_fastmath.exe --n 8192 --steps 50 --repeats 3 --csv results\stage3_sample.csv
results\nbody_tiled_fastmath_um.exe --n 8192 --steps 50 --repeats 3 --csv results\stage3_um_sample.csv
```

`nbody_tiled_fastmath_um` is the same kernel additionally compiled with `--use_fast_math`. On this GPU, the manual intrinsics alone gave a 2-4x speedup over Stage 2 (bigger than Stage 2's own speedup over Stage 1), and `--use_fast_math` added a further, consistent ~7-12% on top, since explicitly calling `rsqrtf` already captured most of the benefit the flag would otherwise provide. Both variants validated correctly against the CPU baseline (see `docs/log.md` for the full numbers).

**Stage 4 (block-size tuning)** is the Stage 3 kernel with block size templated instead of fixed, so a `--block-size` flag can select 64/128/256/512/1024 at runtime while each still gets a fully unrolled inner loop:

```powershell
results\nbody_tuned.exe --n 8192 --steps 50 --repeats 3 --block-size 128 --csv results\stage4_sample.csv
```

To sweep all five block sizes across the full N range in one go:

```bash
./bench/run_sweep.sh
```

Writes `results/stage4_benchmark.csv`. On this GPU, no single block size wins at every N (the best shifts with problem size), but averaged across all N, **block sizes 64 and 128 tie for best overall** (~1.13-1.14x over the best-per-N time), both clearly ahead of 256  -  the default used throughout Stages 1-3  -  which averages 1.50x. Full table and analysis in `docs/log.md`.

Nsight Compute profiling (occupancy, registers/thread, warps/SM) confirms a counterintuitive result: **achieved occupancy rises with block size** (35% at bs=64 up to 67% at bs=1024, which sits right at its own hardware-limited 67% ceiling), yet **wall-clock speed goes the other way**  -  bs=64/128 are fastest despite the lowest occupancy. Registers/thread are constant (40) across all sizes, so this isn't register pressure; it's the classic occupancy-doesn't-equal-performance lesson for a compute-bound kernel, where smaller blocks' extra `__syncthreads()` overhead (more tiles per body) apparently costs less than the occupancy they're missing gains. `ncu` itself required running from an Administrator terminal on this hybrid-graphics laptop (the usual GUI permission grant, NVIDIA Control Panel's Developer Settings, was unavailable since the display isn't driven by the RTX 5050). Full table and commands in `docs/log.md`.

**Stage 5 (Thrust library ceiling)** expresses the same simulation with `thrust::for_each` instead of a hand-written kernel  -  no manual tiling, no launch configuration, and deliberately the plain unoptimized math (matching Stage 1), so it represents the honest "reached for the library instead of hand-tuning" baseline:

```powershell
results\nbody_thrust.exe --n 8192 --steps 50 --repeats 3 --csv results\stage5_sample.csv
```

Same CLI/CSV format as the other stages, defaulting to `results\stage5_benchmark.csv`. On this GPU, the full hand-tuned pipeline (Stage 4) beats this Thrust baseline by **7-11x at small/mid N, narrowing to 2.6x at N=32768** as the problem becomes compute-bound enough to swamp Thrust's abstraction overhead. Comparing Thrust against Stage 1 (hand-written, identical math) isolates that abstraction overhead specifically: ~1.7-1.9x in the mid-size range, converging to near-parity at both extremes. Full tables in `docs/log.md`, including a note on measurement variance across sessions (this laptop GPU's absolute numbers drift somewhat run-to-run, likely thermal/clock related  -  relative comparisons within the same session are the reliable signal).

### Validating a stage against the CPU baseline

Both `nbody_cpu.exe` and `nbody_naive.exe` accept a `--dump-final <path>` flag that writes final body positions to a CSV instead of (in addition to) the timing CSV. Since both share the same seeded RNG for initial conditions, running both with identical `--n`/`--steps` produces directly comparable output, body-for-body, with no matching or sorting needed.

```powershell
results\nbody_cpu.exe --n 4096 --steps 10 --repeats 1 --dump-final results\cpu_final.csv
results\nbody_naive.exe --n 4096 --steps 10 --repeats 1 --warmup 0 --dump-final results\gpu_final.csv
results\validate.exe results\cpu_final.csv results\gpu_final.csv
```

Pass `--warmup 0` to `nbody_naive.exe` when validating: its warmup loop runs real physics steps on the same buffers before the timed section (it is not a throwaway measurement warmup), so it must be zeroed out for the two step counts to match exactly.

Expected output:

```text
bodies=4096 max_dist=... mean_dist=... tolerance=0.01 result=PASS
```

**Use a small step count (10-30) for this check, not a large one.** Gravitational N-body systems are chaotic: a tiny float32-vs-double rounding difference between the CPU baseline (double-precision force accumulation) and the GPU kernel (single-precision throughout) gets exponentially amplified over many steps, especially around close encounters. Measured on this machine at N=4096, max positional deviation grew from `0.00015` at 10 steps to `2.43` at 100 steps, an expected chaotic blowup, not a bug. A short run validates that the force calculation and integration are implemented correctly; a long run will fail regardless of correctness, because no two floating-point implementations of a chaotic system stay in agreement indefinitely.

## Next stages
- Optional contrast kernel (a bandwidth-bound stencil) to show tiling matters more for compute-bound problems than bandwidth-bound ones.
- Visualization: snapshot dumps + offline rendering (not yet implemented; see `project_plan.md`).

## Notes
The current environment exposes CUDA 13.2 and an installed Visual Studio Build Tools toolchain, so the project can be compiled locally for both CPU and CUDA work.
