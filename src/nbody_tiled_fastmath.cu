#include "common.h"

#include <cuda_runtime.h>

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(err) << std::endl;                 \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

// Block size is a generic default; Stage 4 sweeps and tunes this per GPU.
constexpr int kBlockSize = 256;

/* Same shared-memory tiling as Stage 2, plus two changes: the inner tile
loop is unrolled with #pragma unroll, and the 1/sqrt path is replaced with
the rsqrtf intrinsic (a single fast reciprocal-sqrt instruction instead of a
divide following a sqrt) with fmaf used for the accumulation. rsqrtf trades
a small amount of precision for throughput; the accuracy impact is measured
against the CPU baseline separately from Stage 1/2's exact-rounding path. */
__global__ void computeForcesTiledFastKernel(const float4* pos, float4* acc, int n, float epsSq) {
  __shared__ float4 tile[kBlockSize];

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float4 pi = (i < n) ? pos[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;

  const int numTiles = (n + kBlockSize - 1) / kBlockSize;
  for (int t = 0; t < numTiles; ++t) {
    const int j = t * kBlockSize + threadIdx.x;
    tile[threadIdx.x] = (j < n) ? pos[j] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < kBlockSize; ++k) {
      const float4 pj = tile[k];
      const float dx = pj.x - pi.x;
      const float dy = pj.y - pi.y;
      const float dz = pj.z - pi.z;
      const float distSq = dx * dx + dy * dy + dz * dz + epsSq;
      const float invDist = rsqrtf(distSq);
      const float invDistCubed = invDist * invDist * invDist;
      const float strength = pj.w * invDistCubed;

      ax = fmaf(strength, dx, ax);
      ay = fmaf(strength, dy, ay);
      az = fmaf(strength, dz, az);
    }
    __syncthreads();
  }

  if (i < n) {
    acc[i] = make_float4(ax, ay, az, 0.0f);
  }
}

// Semi-implicit Euler: velocity updates from the acceleration computed above,
// then position updates from the new velocity. Output-centric, no atomics.
// Identical to Stage 1/2: integration is O(N), not what this stage targets.
__global__ void integrateKernel(float4* pos, float4* vel, const float4* acc, int n, float dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  float4 v = vel[i];
  v.x += acc[i].x * dt;
  v.y += acc[i].y * dt;
  v.z += acc[i].z * dt;
  vel[i] = v;

  float4 p = pos[i];
  p.x += v.x * dt;
  p.y += v.y * dt;
  p.z += v.z * dt;
  pos[i] = p;
}

int main(int argc, char** argv) {
  SimConfig config = defaultConfig();
  std::string csvPath = "results/stage3_benchmark.csv";
  std::string dumpFinalPath;
  bool fullSweep = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--n" && i + 1 < argc) {
      config.numBodies = std::stoi(argv[++i]);
    } else if (arg == "--steps" && i + 1 < argc) {
      config.steps = std::stoi(argv[++i]);
    } else if (arg == "--repeats" && i + 1 < argc) {
      config.repeats = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      config.warmup = std::stoi(argv[++i]);
    } else if (arg == "--csv" && i + 1 < argc) {
      csvPath = argv[++i];
    } else if (arg == "--dump-final" && i + 1 < argc) {
      dumpFinalPath = argv[++i];
    } else if (arg == "--sweep") {
      fullSweep = true;
    } else if (arg == "--colliding") {
      config.useCollidingClusters = true;
    }
  }

  std::vector<int> sizes;
  if (fullSweep) {
    sizes = {1024, 2048, 4096, 8192, 16384, 32768};
  } else {
    sizes.push_back(config.numBodies);
  }

  const std::vector<std::string> header = {"n", "steps", "repeats", "avg_ms_per_step", "avg_total_ms"};
  writeCsvHeader(csvPath, header);

  for (int n : sizes) {
    const int blocks = (n + kBlockSize - 1) / kBlockSize;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float4);

    float4* dPos = nullptr;
    float4* dVel = nullptr;
    float4* dAcc = nullptr;
    CUDA_CHECK(cudaMalloc(&dPos, bytes));
    CUDA_CHECK(cudaMalloc(&dVel, bytes));
    CUDA_CHECK(cudaMalloc(&dAcc, bytes));

    std::vector<double> runTimes;
    runTimes.reserve(config.repeats);

    for (int run = 0; run < config.repeats; ++run) {
      std::vector<float4> hPos(static_cast<size_t>(n));
      std::vector<float4> hVel(static_cast<size_t>(n));
      initializeBodies(hPos, hVel, n, config.seed + run, config.useCollidingClusters);

      CUDA_CHECK(cudaMemcpy(dPos, hPos.data(), bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dVel, hVel.data(), bytes, cudaMemcpyHostToDevice));

      for (int step = 0; step < config.warmup; ++step) {
        computeForcesTiledFastKernel<<<blocks, kBlockSize>>>(dPos, dAcc, n, config.epsilonSquared);
        integrateKernel<<<blocks, kBlockSize>>>(dPos, dVel, dAcc, n, config.dt);
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      const double start = nowMs();
      for (int step = 0; step < config.steps; ++step) {
        computeForcesTiledFastKernel<<<blocks, kBlockSize>>>(dPos, dAcc, n, config.epsilonSquared);
        integrateKernel<<<blocks, kBlockSize>>>(dPos, dVel, dAcc, n, config.dt);
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      runTimes.push_back(nowMs() - start);

      if (run == 0 && !dumpFinalPath.empty()) {
        std::vector<float4> finalPos(static_cast<size_t>(n));
        CUDA_CHECK(cudaMemcpy(finalPos.data(), dPos, bytes, cudaMemcpyDeviceToHost));
        writePositionsCsv(dumpFinalPath, finalPos);
      }
    }

    CUDA_CHECK(cudaFree(dPos));
    CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaFree(dAcc));

    const double avgTotalMs = std::accumulate(runTimes.begin(), runTimes.end(), 0.0) / static_cast<double>(runTimes.size());
    const double avgPerStepMs = avgTotalMs / static_cast<double>(config.steps);

    appendCsvRow(csvPath, {
        std::to_string(n),
        std::to_string(config.steps),
        std::to_string(config.repeats),
        std::to_string(avgPerStepMs),
        std::to_string(avgTotalMs)
    });

    std::cout << "n=" << n
              << " avg_ms_per_step=" << avgPerStepMs
              << " avg_total_ms=" << avgTotalMs << '\n';
  }

  return 0;
}
