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

## Next stages
- Stage 2: shared-memory tiling.
- Stage 3: loop unrolling and fast-math.
- Stage 4: block-size and occupancy tuning.

## Notes
The current environment exposes CUDA 13.2 and an installed Visual Studio Build Tools toolchain, so the project can be compiled locally for both CPU and CUDA work.
