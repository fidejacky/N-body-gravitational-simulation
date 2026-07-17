# Direct N-Body Simulation in CUDA

A hand-written CUDA implementation of a direct (all-pairs) gravitational N-body
simulation, built to explore how a compute-bound problem with high arithmetic
intensity maps onto the GPU. This is a school project for a GPU Computing
course, and the work is organized as an iterative set of independently
benchmarkable stages. The project starts from a serial CPU baseline and stacks
five optimizations  -  each measured independently  -  up to a tiled,
occupancy-tuned kernel, with a library-based implementation used only as a
performance ceiling for comparison.

Before choosing hardware-specific defaults such as block size, occupancy targets,
or fast-math settings, the target GPU and CUDA toolkit version should be
confirmed so the implementation remains portable and the reported results are
meaningful.

The core kernels are written from scratch; Thrust and cuBLAS appear only for
validation and as a reference upper bound, not as the main implementation.

## Background

The simulation tracks `N` bodies, each with a position, velocity, and mass.
Every body feels a gravitational pull from every other body. Each timestep
computes the net force on each body  -  an O(N²) operation, since every body
interacts with every other  -  then updates velocities and positions with a
simple integrator.

This is the *direct* or *all-pairs* method, which evaluates every interaction
explicitly rather than approximating distant groups (as Barnes-Hut does). The
brute-force version is chosen deliberately: it is the variant that maps cleanly
onto the GPU and exposes a clear optimization path. Parallelization is
output-centric  -  one thread owns one body and accumulates the total force acting
on it  -  so no atomics are needed, because each thread writes only its own result.

The reason this problem suits the GPU is its high arithmetic intensity: a large
number of floating-point operations per byte loaded, since each body's data is
reused against all other `N` bodies. As a result, shared-memory tiling and
occupancy tuning produce real speedups, in contrast to memory-bandwidth-bound
problems where such optimizations have limited effect.

## Physics

The acceleration on body *i* from all other bodies *j* is:

```
a_i = G * Σ (over j≠i)  m_j * (r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2)
```

- `G` is the gravitational constant, set to 1.0 in simulation units (physical
  realism is not a goal of the project).
- `ε` is a softening factor  -  a small constant added to the denominator so that
  forces between very close bodies stay finite instead of diverging and
  destabilizing the simulation. `ε²` is set in the range 0.01–0.1.
- The `^(3/2)` term is evaluated as a reciprocal square root followed by cubing,
  which is where the fast-math intrinsic optimization is applied later.

A semi-implicit Euler integrator updates the state each step:

```
v_i += a_i * dt
x_i += v_i * dt
```

with a small `dt` (around 0.01). The integration scheme is not the focus of the
project, so a simple integrator is sufficient.

## Data layout

Memory layout is treated as an explicit optimization lever. Two options were
considered:

- **Array of Structs (AoS):** `struct Body { float4 pos; float4 vel; }`, stored
  as `Body bodies[N]`. Intuitive, but leads to strided, uncoalesced memory
  access.
- **Struct of Arrays (SoA):** separate `float4* pos` and `float4* vel` arrays,
  giving coalesced access where adjacent threads read adjacent memory.

The implementation uses SoA. Following the approach from the classic GPU Gems
N-body chapter, position and mass are packed into a single `float4` (`x, y, z,
mass`), so a single aligned 16-byte load retrieves everything needed for one
body. Velocity is stored as a `float4` as well. This packing is a deliberate
coalescing and alignment optimization.

## Optimization stages

Each stage is a separate, independently benchmarked version, and all versions
remain runnable so results can be regenerated.

### Stage 0  -  Serial CPU baseline

A plain C++ implementation with a double-nested loop over all pairs, compiled
with `-O3`. This provides the reference time against which all GPU speedups are
measured.

### Stage 1  -  Naive GPU kernel

One thread per body. Each thread loops over all `N` bodies, reading their
positions directly from global memory and accumulating acceleration.
Output-centric with no atomics. Every thread re-reads all `N` positions from
global memory, producing N× redundant global traffic  -  the bottleneck the next
stage targets.

### Stage 2  -  Shared-memory tiling

Each block cooperatively loads a tile of body positions into shared memory; every
thread in the block computes interactions against that tile, then synchronizes
and loads the next tile. Each body's data is loaded from global memory once per
block rather than once per thread. The tile width equals the block size. This is
the canonical N-body tiling pattern, where a `p × p` block of interactions is
evaluated from `p` loaded bodies, and it is where the problem's high arithmetic
intensity is exploited.

### Stage 3  -  Loop unrolling and fast-math intrinsics

Two changes: `#pragma unroll` on the inner tile loop, and replacing the
`pow(..., 1.5)` / `sqrt` path with the `rsqrtf` intrinsic and explicit
multiplies, using `fmaf` for the multiply-adds. The fast-math path can also be
compiled with `-use_fast_math`, with the resulting accuracy difference recorded
against the double-precision CPU baseline.

### Stage 4  -  Block-size and occupancy tuning

A sweep over block sizes (64, 128, 256, 512, 1024), with achieved occupancy,
registers per thread, and warps per SM read from Nsight Compute. The best
configuration is selected and explained in terms of occupancy and register
pressure.

### Stage 5  -  Library ceiling

The same simulation expressed through a Thrust-based formulation (or compared
against a reference implementation such as the CUDA samples `nbody` binary),
used as a performance ceiling. The hand-tuned kernel is reported as a fraction
of the optimized reference's performance.

### Optional contrast kernel

If time allows, a memory-bandwidth-bound kernel (such as a simple stencil) is
added to show that tiling produces a large speedup on N-body but only a small
one on the bandwidth-bound kernel  -  demonstrating *when* these GPU optimizations
pay off, not just that they do.

## Measurement methodology

- Kernel timing uses CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`)
  rather than host-side wall clock alone.
- A few untimed warm-up iterations run before measurement to exclude one-time
  setup costs.
- Each configuration runs a fixed number of steps (100), reported as milliseconds
  per step.
- Each measurement is repeated several times and reported as the mean across
  runs.
- Simulation data stays resident on the GPU across all timesteps; host transfers
  happen only when snapshots are taken for visualization, avoiding the
  per-iteration transfer overhead that would otherwise dominate runtime.

Problem sizes swept: N = 1024, 2048, 4096, 8192, 16384, 32768, and higher where
it fits and completes. A log-log plot of time versus N shows the O(N²) slope and
the widening gap between CPU and GPU as N grows.

Nsight Compute metrics captured for the profiling section: achieved occupancy,
registers per thread, compute (SM) throughput, memory throughput, and warps per
SM. The combination of high SM throughput and lower memory throughput is the
direct evidence that the kernel is compute-bound.

Correctness is validated by comparing GPU final positions against the CPU
baseline within a tolerance that accounts for the float-versus-double difference.

## Visualization

Positions are dumped to disk every K steps (K = 10) as the simulation runs. An
offline Python script (matplotlib) reads the dump and renders each snapshot as a
scatter plot, and the frames are stitched into a GIF or MP4 with `imageio` or
`ffmpeg`. Initial conditions seed two clusters with opposing bulk velocities (a
"galaxy collision") or a rotating disk, which produce visually clear structure.
Visualization is intentionally kept offline and lightweight rather than using
real-time OpenGL/CUDA interop.

## Repository structure

```
nbody-cuda/
  src/
    nbody_cpu.cpp            # Stage 0
    nbody_naive.cu          # Stage 1
    nbody_tiled.cu          # Stage 2
    nbody_tiled_fastmath.cu # Stage 3 (or a compile flag on Stage 2)
    common.h                # float4 helpers, init, I/O, timing
  bench/
    run_sweep.sh            # sweeps N and block sizes, logs CSV
  viz/
    render.py               # snapshots -> PNG -> GIF
  results/
    timings.csv
    ncu_reports/
  README.md
```

Stages are kept as separate files (or guarded by `#ifdef`) so any version can be
rebuilt on demand.

## Notes

- The CPU baseline is compiled with `-O3` so that reported speedups reflect a
  fair comparison.
- The softening factor is required; without it, close encounters produce
  diverging forces and an unstable simulation.
- GPU computation runs in single precision while the CPU baseline uses double
  precision, so small numerical divergence is expected and reported as a
  tolerance rather than treated as exact agreement.
- Results are reported across the full range of N rather than as a single
  number, since the widening CPU/GPU gap is the most informative part of the
  comparison.