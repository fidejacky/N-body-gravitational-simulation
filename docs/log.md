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
