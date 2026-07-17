# Files

- `nbody_cpu.cpp`: CPU baseline, no optimization.
- `nbody_naive.cu`: one thread per body.
- `nbody_tiled.cu`: shared-memory tiling.
- `nbody_tiled_fastmath.cu`: loop unrolling + rsqrtf/fmaf.
- `nbody_tuned.cu`: block-size tuning.
- `nbody_thrust.cu`: Thrust library ceiling.
- `common.h`: shared helpers, no kernel.
- `validate.cpp`: CPU-vs-GPU correctness check.
