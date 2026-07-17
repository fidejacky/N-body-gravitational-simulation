# Files

- `nbody_cpu.cpp`: CPU baseline, no optimization.
- `nbody_tuned.cu`: one thread per body, shared-memory tiling, loop unrolling with rsqrtf/fmaf intrinsics, and runtime block-size tuning.
- `common.h`: shared helpers (float4 math, init, timing, CSV), required by both files above.
- `CMakeLists.txt`: build configuration for both executables.
- `build.ps1`: one-command build script, Windows.
- `build.sh`: one-command build script, Linux/macOS.

# How to compile

Prerequisites: CMake 3.24+, Ninja, a CUDA toolkit with `nvcc`, and (on Windows) MSVC Build Tools.

Windows:
```
.\build.ps1
```

Linux/macOS:
```
./build.sh
```

Either script configures and builds both executables in one step, writing them to `results/`. Under the hood it runs:
```
nvcc -O2 -std=c++17 src\nbody_cpu.cpp -o results\nbody_cpu.exe
nvcc -O2 -std=c++17 -arch=sm_120 src\nbody_tuned.cu -o results\nbody_tuned.exe
```
`-arch=sm_120` targets this GPU's compute capability (RTX 5050, Blackwell)
