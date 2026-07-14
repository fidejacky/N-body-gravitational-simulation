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
- [src/common.h](src/common.h) — shared helpers for float4 operations, initialization, timing, and CSV output.
- [src/nbody_cpu.cpp](src/nbody_cpu.cpp) — Stage 0 CPU baseline.
- [src/nbody_naive.cu](src/nbody_naive.cu) — Stage 1 CUDA scaffold.
- [docs/log.md](docs/log.md) — iterative development log.
- [project_plan.md](project_plan.md) — project plan and stage breakdown.
- [CLAUDE.md](CLAUDE.md) — course-project context and current constraints.

## Build

Prerequisites (one-time per machine): [CMake](https://cmake.org/) 3.24+, [Ninja](https://ninja-build.org/), a CUDA toolkit with `nvcc`, and MSVC Build Tools on Windows.

```powershell
# Windows
.\build.ps1
```

```bash
# Linux/macOS
./build.sh
```

Both scripts configure and build every stage (`nbody_cpu`, `nbody_naive`, ...) with a single command via CMake + Ninja, writing binaries to `results/`. On Windows, `build.ps1` locates and loads the MSVC developer environment automatically via `vswhere`, so there's no need to open a Developer Command Prompt or run `vcvars64.bat` manually.

Run a stage directly, e.g.:

```powershell
results\nbody_cpu.exe --n 1024 --steps 5 --repeats 1 --csv results\stage0_sample.csv
```

## Next stages
- Stage 1: naive GPU kernel with one thread per body.
- Stage 2: shared-memory tiling.
- Stage 3: loop unrolling and fast-math.
- Stage 4: block-size and occupancy tuning.

## Notes
The current environment exposes CUDA 13.2 and an installed Visual Studio Build Tools toolchain, so the project can be compiled locally for both CPU and CUDA work.
