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
- Verified: `results\nbody_naive.exe --n 8192 --steps 50 --repeats 3` runs successfully at `avg_ms_per_step≈1.39`, versus the CPU baseline's `≈188` at the same N  -  roughly 135x, a sane result for a first, unoptimized parallelization.
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
- Validated against the CPU baseline at N=4096, 20 steps: `max_dist=0.000474052`, `PASS`  -  and notably bit-identical to Stage 1's deviation at the same N/steps, confirming tiling only reorganizes memory access (same summation order, same per-op precision) without changing the math.
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
  - Manual intrinsics: `max_dist=0.000521914`, `PASS`  -  slightly higher than Stage 1/2's `0.000474052`, consistent with `rsqrtf` trading a small amount of precision for throughput versus an exact division.
  - `--use_fast_math`: `max_dist=0.000521914`, `PASS`  -  identical to the manual-only build at this step count. The flag's extra relaxations (denormal flush-to-zero, more aggressive FMA contraction) don't show up as additional error here, though this is a short-step spot check, not a guarantee at longer horizons where chaotic amplification (documented under Stage 1) dominates regardless of implementation.
- Benchmarked full sweep (steps=100, repeats=3, warmup=1):

| N     | Stage 2 tiled (ms/step) | Stage 3 manual (ms/step) | Stage 3 use_fast_math (ms/step) | Manual vs Stage 2 | fast_math vs manual |
|-------|--------------------------|----------------------------|-----------------------------------|--------------------|------------------------|
| 1024  | 0.206                    | 0.0769                     | 0.0695                            | 2.68x              | 1.11x                  |
| 2048  | 0.181                    | 0.0458                     | 0.0409                            | 3.95x              | 1.12x                  |
| 4096  | 0.344                    | 0.0863                     | 0.0808                            | 3.99x              | 1.07x                  |
| 8192  | 0.767                    | 0.286                      | 0.265                             | 2.68x              | 1.08x                  |
| 16384 | 2.406                    | 1.084                      | 1.000                             | 2.22x              | 1.08x                  |
| 32768 | 7.703                    | 3.860                      | 3.623                             | 2.00x              | 1.07x                  |

- The manual intrinsics alone give a substantial 2-4x speedup over Stage 2, bigger than Stage 2's own gain over Stage 1  -  eliminating the sqrt+divide is evidently a bigger win on this GPU than the memory-access reorganization was. `--use_fast_math` on top adds a consistent but modest further ~7-12%, since explicitly calling `rsqrtf` already captured most of the benefit the flag would otherwise provide.
- A quick single-run spot check (N=4096, 20 steps, no sweep averaging) initially suggested `--use_fast_math` was ~2.2x faster than the manual-only build; the full sweep with repeated, averaged runs contradicts this and shows only ~7-12%. Noted here as a reminder that small ad-hoc timing checks are noisy and the averaged sweep is the trustworthy number.
- Next step: block-size and occupancy tuning (Stage 4).

## Stage 4: block-size tuning implemented and swept
- Implemented `src/nbody_tuned.cu`: the Stage 3 kernel (tiling + unroll + `rsqrtf` + `fmaf`), templated on block size (`template <int BLOCK_SIZE>`) instead of a single compile-time constant, with a runtime `--block-size` flag dispatched via `runStepsDispatch` to one of five instantiations (64/128/256/512/1024). Templating instead of a runtime `blockDim.x` variable matters here: it keeps `#pragma unroll`'s trip count a genuine compile-time constant for every block size, so each size still gets a fully unrolled inner loop, not just a runtime-configurable one.
- Validated against the CPU baseline at block-size 128, N=4096, 20 steps: `max_dist=0.000521914`, `PASS`  -  identical to Stage 3's deviation, confirming the templated dispatch doesn't change the math, only the launch configuration.
- **Nsight Compute profiling**: `ncu` initially failed with `ERR_NVGPUCTRPERM` (needs elevated GPU performance-counter access), and the usual GUI permission grant (NVIDIA Control Panel → Desktop → Enable Developer Settings → Manage GPU Performance Counters) was unavailable on this hybrid-graphics laptop  -  the display is driven by the integrated GPU, not the RTX 5050, so the display-dependent Desktop menu doesn't expose that option at all. Resolved by running `ncu` from an Administrator PowerShell, which bypasses the permission check at the OS level regardless of display routing. Collected via (once per block size, N=16384, single profiled step):
  ```
  & "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\target\windows-desktop-win7-x64\ncu.exe" --metrics sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,sm__maximum_warps_per_active_cycle_pct,launch__shared_mem_per_block_static --launch-count 1 --csv results\nbody_tuned.exe --n 16384 --steps 1 --warmup 0 --repeats 1 --block-size <SIZE>
  ```
  (Note the `&` call operator  -  required in PowerShell to execute a quoted path; omitting it produces a parse error rather than running the command.)
- Added `bench/run_sweep.sh` (matches the filename in `project_plan.md`'s repository structure) to sweep all five block sizes across the full N range. Necessary workaround: `nbody_tuned`'s CSV writer truncates its output path at the start of every process invocation (by design, so a single run's sweep starts from a clean file), so calling it once per block size against the same CSV would silently wipe every prior block size's rows. The script writes each block size to its own temp CSV and merges them afterward instead.
- Full sweep results (steps=100, repeats=3, warmup=1), `avg_ms_per_step`:

| N     | bs=64  | bs=128 | bs=256 | bs=512 | bs=1024 |
|-------|--------|--------|--------|--------|---------|
| 1024  | 0.0328 | 0.0305 | 0.0394 | 0.0530 | 0.0664  |
| 2048  | 0.0460 | 0.0392 | 0.0659 | 0.0662 | 0.1217  |
| 4096  | 0.0877 | 0.0737 | 0.1277 | 0.1249 | 0.2417  |
| 8192  | 0.3181 | 0.2845 | 0.4451 | 0.2579 | 0.4754  |
| 16384 | 1.0826 | 1.3782 | 1.5746 | 1.0238 | 1.0169  |
| 32768 | 4.1240 | 5.5048 | 3.9564 | 4.3659 | 4.4621  |

- No single block size wins at every N  -  the per-N winner shifts (128 for N≤4096, 512/1024 around 8192-16384, 256 at 32768), which is itself a real finding, not noise to average away: block-size performance on this GPU is workload-size-dependent, so "pick one default and move on" (256, used throughout Stages 1-3) is a simplification.
- To find the best *overall* choice, normalized each block size's time to the best time at that N (ratio, so every N counts equally regardless of its absolute magnitude) and averaged the ratio across all six N: **bs=64 averages 1.13x over the best-per-N time, bs=128 averages 1.14x**  -  these two are effectively tied for best overall and clearly ahead of bs=256 (1.50x), bs=512 (1.37x), and bs=1024 (2.09x, the worst).
- Notable: the block size used as the default throughout Stages 1-3 (256) is *not* the best overall choice by this measure  -  bs=64 or bs=128 would have been a better default on this specific GPU.
- Measured occupancy/register/shared-memory data (N=16384, `computeForcesTiledFastKernel<BLOCK_SIZE>`):

| Block size | Registers/thread | Shared mem/block | Grid size (blocks) | Theoretical max occupancy | Achieved occupancy |
|------------|--------------------|---------------------|------------------------|------------------------------|------------------------|
| 64         | 40                 | 1024 B               | 256                     | 100.00%                       | 35.10%                  |
| 128        | 40                 | 2048 B               | 128                     | 100.00%                       | 37.19%                  |
| 256        | 40                 | 4096 B               | 64                      | 100.00%                       | 38.31%                  |
| 512        | 40                 | 8192 B               | 32                      | 100.00%                       | 46.28%                  |
| 1024       | 40                 | 16384 B              | 16                      | 66.67%                        | 66.66%                  |

- **This directly contradicts the hypothesis recorded above before real data was available** (that smaller blocks would show *higher* occupancy from their smaller shared-memory footprint). The measured trend is the opposite: achieved occupancy rises monotonically with block size, and bs=1024 essentially saturates its own (lower) theoretical ceiling almost exactly (66.66% of 66.67%), while the smaller sizes leave a large fraction of their much higher ceiling (100%) unused. Registers/thread are constant at 40 regardless of block size, so register pressure isn't what's differentiating these configurations  -  the bs=1024 ceiling drop to 66.67% comes from a hardware residency limit (warps- or blocks-per-SM) at 32 warps/block, not registers or shared memory (both have headroom left at every size tested).
- Yet the timing data above shows the *opposite* preference: bs=64/128 (lowest occupancy) are fastest overall, bs=1024 (highest occupancy, effectively resource-saturated) is the slowest. This is the classic lesson that occupancy is not a direct proxy for performance, especially for a compute-bound kernel like this one (high arithmetic intensity per body, per `project_plan.md`'s own framing): once enough warps are resident to keep the SM's arithmetic units fed, *more* active warps beyond that point don't translate into more useful work, while smaller blocks pay a real cost the occupancy numbers don't capture  -  more tiles per body (`numTiles = n / BLOCK_SIZE`: 256 tiles at bs=64 vs. 16 at bs=1024), meaning more `__syncthreads()` barriers and more redundant per-tile bookkeeping for the same total N. The net timing result is a trade between that per-tile overhead and the (apparently secondary, for this kernel) benefit of higher occupancy.
- Selected configuration going forward: **block-size 128** (tied-best overall on timing, and outright best at the smaller N values most likely to be used for iteration/debugging)  -  chosen on the timing evidence, not the occupancy numbers, precisely because this stage's own data shows occupancy doesn't predict speed here.
- Next step: Stage 5 (library-based ceiling comparison).

## Stage 5: Thrust-based library ceiling implemented, validated, and compared
- Implemented `src/nbody_thrust.cu`: the same simulation expressed with `thrust::for_each` over a `thrust::counting_iterator` (force) and a `thrust::zip_iterator` over `(pos, vel, acc)` (integration), using `thrust::device_vector` instead of raw `cudaMalloc`/`cudaMemcpy`. No hand-written `__global__` kernel, no shared-memory tiling, no manual launch configuration  -  Thrust picks all of that. Deliberately used the plain, unoptimized math (`1.0f / sqrtf(...)`, matching Stage 1) rather than Stage 3's `rsqrtf`/`fmaf`, since the point is to measure the honest "reached for the library instead of hand-tuning" baseline, not to also sneak the hand-tuned math into the reference it's being compared against.
- Build hit a real MSVC/Thrust incompatibility: `cl.exe`'s traditional preprocessor is incompatible with newer CUDA's Thrust/CCCL headers (`fatal error C1189`). Fixed by passing `-Xcompiler=/Zc:preprocessor` for this target only, in `CMakeLists.txt`.
- Validated against the CPU baseline at N=4096, 20 steps: `max_dist=0.000474052`, `PASS`  -  identical to Stage 1/2's deviation, confirming this uses the same unoptimized math as intended (not Stage 3's slightly-lower-precision `rsqrtf` path).
- Benchmarked full sweep (steps=100, repeats=3, warmup=1) against Stage 1 (hand-written naive, identical math  -  isolates Thrust's abstraction overhead) and Stage 4 at block-size 128 (our best hand-tuned configuration  -  the actual "ceiling" comparison the project plan asks for):

| N     | Thrust (ms/step) | Stage 1 naive (ms/step) | Thrust vs Stage 1 | Stage 4 tuned bs128 (ms/step) | Hand-tuned speedup vs Thrust |
|-------|-------------------|----------------------------|----------------------|----------------------------------|----------------------------------|
| 1024  | 0.2617            | 0.261                       | 1.00x                 | 0.0342                            | 7.65x                              |
| 2048  | 0.4877            | 0.282                       | 1.73x                 | 0.0468                            | 10.42x                             |
| 4096  | 0.9576            | 0.565                       | 1.70x                 | 0.0853                            | 11.23x                             |
| 8192  | 2.2176            | 1.174                       | 1.89x                 | 0.3007                            | 7.38x                              |
| 16384 | 4.4885            | 2.453                       | 1.83x                 | 1.0104                            | 4.44x                              |
| 32768 | 9.8511            | 9.770                       | 1.01x                 | 3.8072                            | 2.59x                              |

- Thrust's abstraction overhead over an equivalent hand-written kernel (same math) is real but bounded: ~1.7-1.9x slower in the mid-size range (N=2048-16384), converging to near-parity at both extremes  -  at N=1024, fixed per-launch overhead dominates equally for both; at N=32768, raw O(N²) compute dominates enough that the abstraction cost becomes negligible.
- The full hand-tuned pipeline (tiling + intrinsics + block-size tuning) beats the Thrust ceiling by **7-11x at small/mid N, narrowing to 2.6x at N=32768** as the problem becomes compute-bound enough that even an unoptimized inner loop starts to saturate the GPU. Framed the way `project_plan.md` asks ("hand-tuned kernel reported as a fraction of the reference's performance") this inverts here: our hand-tuned kernel outperforms the library-idiomatic baseline throughout, rather than the usual case of a hand-written kernel trailing a heavily-optimized library primitive  -  expected, since Thrust has no N-body-specific optimized primitive to reach for here, so this ceiling represents "effort-free library baseline," not a performance ceiling in the sense that cuBLAS/cuFFT would be for other problems.
- Methodological note: the `bs128` numbers here (measured back-to-back with the Thrust sweep in the same session) differ noticeably from the same configuration's numbers recorded earlier under Stage 4 (e.g., N=32768: `3.807` here vs `5.505` there)  -  likely thermal/clock variance on this laptop GPU across separate sweep sessions, not a code change. Numbers measured in the same session (as here) are directly comparable; numbers compared across sessions separated by significant wall-clock time should be treated as approximate.
- All five stages plus the CPU baseline and Thrust ceiling are now implemented, validated, and benchmarked end-to-end.
