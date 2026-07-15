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

## Baseline sweep recorded (Stage 0 and Stage 1)

Full `--sweep` (N = 1024..32768, steps=100, repeats=3, warmup=1 for Stage 1) recorded before starting Stage 2, so its improvement has a clean "before" to compare against:

| N     | Stage 0 CPU (ms/step) | Stage 1 naive GPU (ms/step) | Speedup |
|-------|------------------------|------------------------------|---------|
| 1024  | 2.87                   | 0.261                        | 11x     |
| 2048  | 11.4                   | 0.282                        | 40x     |
| 4096  | 69.2                   | 0.565                        | 122x    |
| 8192  | 227.5                  | 1.174                        | 194x    |
| 16384 | 728.5                  | 2.453                        | 297x    |
| 32768 | 3149.3                 | 9.770                        | 322x    |

Confirms the O(N²) slope on the CPU side and a widening CPU/GPU gap as N grows, as expected. Stage 0 numbers are in `results/stage0_benchmark.csv`, Stage 1 in `results/stage1_benchmark.csv`.

## Stage 2: shared-memory tiling implemented, validated, and benchmarked
- Implemented `computeForcesTiledKernel` in `src/nbody_tiled.cu`: each block cooperatively loads one `kBlockSize`-wide tile of body positions into `__shared__` memory, every thread in the block computes against that tile, then the block `__syncthreads()`s and loads the next tile. Cuts global memory reads from N-per-thread (Stage 1) to N-per-block. Threads past `n` in the last partial block still participate in every `__syncthreads()` with a dummy zero-mass body, since all threads in a block must hit synchronization points together; they just skip the final `acc[]` write. `integrateKernel` is unchanged from Stage 1 (integration is O(N), not what tiling targets).
- CLI/CSV/`--dump-final` harness is identical to Stage 1's, just pointed at `results/stage2_benchmark.csv` by default.
- Validated against the CPU baseline at N=4096, 20 steps: `max_dist=0.000474052`, `PASS` — and notably bit-identical to Stage 1's deviation at the same N/steps, confirming tiling only reorganizes memory access (same summation order, same per-op precision) without changing the math.
- Benchmarked full sweep (steps=100, repeats=3, warmup=1) against Stage 1:

| N     | Stage 1 naive (ms/step) | Stage 2 tiled (ms/step) | Speedup vs Stage 1 |
|-------|--------------------------|--------------------------|---------------------|
| 1024  | 0.261                    | 0.206                    | 1.27x               |
| 2048  | 0.282                    | 0.181                    | 1.56x               |
| 4096  | 0.565                    | 0.344                    | 1.64x               |
| 8192  | 1.174                    | 0.767                    | 1.53x               |
| 16384 | 2.453                    | 2.406                    | 1.02x               |
| 32768 | 9.770                    | 7.703                    | 1.27x               |

- Speedup is real but modest (1.0-1.6x), smaller than the canonical tiling result in the literature. Plausible reason: this GPU's L2 cache is large relative to these problem sizes, so Stage 1's redundant global reads were likely already being served from L2 rather than DRAM, dampening the benefit of explicit shared-memory reuse. Worth revisiting with Nsight Compute memory-throughput metrics in the profiling section (Stage 4).
- Next step: loop unrolling and fast-math intrinsics (Stage 3).

## Stage 3: loop unrolling and fast-math intrinsics implemented, validated, and benchmarked
- Implemented `computeForcesTiledFastKernel` in `src/nbody_tiled_fastmath.cu`: same shared-memory tiling as Stage 2, plus `#pragma unroll` on the inner tile loop, `rsqrtf` replacing the `1.0f / sqrtf(...)` path, and `fmaf` for the three acceleration accumulations. `integrateKernel` unchanged (integration is O(N), not this stage's target).
- Built as **two** CMake targets from the same source, to isolate the compiler flag's effect from the manual intrinsics: `nbody_tiled_fastmath` (manual changes only) and `nbody_tiled_fastmath_um` (same source, additionally compiled with `--use_fast_math`), per `project_plan.md`'s call to record the accuracy difference from `-use_fast_math` separately.
- Validated both against the CPU baseline at N=4096, 20 steps:
  - Manual intrinsics: `max_dist=0.000521914`, `PASS` — slightly higher than Stage 1/2's `0.000474052`, consistent with `rsqrtf` trading a small amount of precision for throughput versus an exact division.
  - `--use_fast_math`: `max_dist=0.000521914`, `PASS` — identical to the manual-only build at this step count. The flag's extra relaxations (denormal flush-to-zero, more aggressive FMA contraction) don't show up as additional error here, though this is a short-step spot check, not a guarantee at longer horizons where chaotic amplification (documented under Stage 1) dominates regardless of implementation.
- Benchmarked full sweep (steps=100, repeats=3, warmup=1):

| N     | Stage 2 tiled (ms/step) | Stage 3 manual (ms/step) | Stage 3 use_fast_math (ms/step) | Manual vs Stage 2 | fast_math vs manual |
|-------|--------------------------|----------------------------|-----------------------------------|--------------------|------------------------|
| 1024  | 0.206                    | 0.0769                     | 0.0695                            | 2.68x              | 1.11x                  |
| 2048  | 0.181                    | 0.0458                     | 0.0409                            | 3.95x              | 1.12x                  |
| 4096  | 0.344                    | 0.0863                     | 0.0808                            | 3.99x              | 1.07x                  |
| 8192  | 0.767                    | 0.286                      | 0.265                             | 2.68x              | 1.08x                  |
| 16384 | 2.406                    | 1.084                      | 1.000                             | 2.22x              | 1.08x                  |
| 32768 | 7.703                    | 3.860                      | 3.623                             | 2.00x              | 1.07x                  |

- The manual intrinsics alone give a substantial 2-4x speedup over Stage 2, bigger than Stage 2's own gain over Stage 1 — eliminating the sqrt+divide is evidently a bigger win on this GPU than the memory-access reorganization was. `--use_fast_math` on top adds a consistent but modest further ~7-12%, since explicitly calling `rsqrtf` already captured most of the benefit the flag would otherwise provide.
- A quick single-run spot check (N=4096, 20 steps, no sweep averaging) initially suggested `--use_fast_math` was ~2.2x faster than the manual-only build; the full sweep with repeated, averaged runs contradicts this and shows only ~7-12%. Noted here as a reminder that small ad-hoc timing checks are noisy and the averaged sweep is the trustworthy number.
- Next step: block-size and occupancy tuning (Stage 4).
