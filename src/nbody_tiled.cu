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
// The tile width equals the block size, so this also sets the shared-memory
// tile size below.
constexpr int kBlockSize = 256;

/* Each block cooperatively loads one tile of kBlockSize bodies into shared
memory at a time, and every thread in the block computes its own body's
interactions against that whole tile before the block moves to the next one.
This turns what was N global-memory reads per thread (Stage 1) into N reads
per BLOCK, cutting global traffic by a factor of kBlockSize.

Threads whose global index is out of range (last, partial block) still have
to participate in every __syncthreads(), since all threads in a block must
reach the same synchronization points; they load a dummy body (mass 0, so it
contributes nothing to anyone's sum, matching the padding at the tail end of
n) and simply skip the final write to acc[]. */
__global__ void computeForcesTiledKernel(const float4* pos, float4* acc, int n, float epsSq) {
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

    for (int k = 0; k < kBlockSize; ++k) {
      const float4 pj = tile[k];
      const float dx = pj.x - pi.x;
      const float dy = pj.y - pi.y;
      const float dz = pj.z - pi.z;
      const float distSq = dx * dx + dy * dy + dz * dz + epsSq;
      const float invDist = 1.0f / sqrtf(distSq);
      const float invDistCubed = invDist * invDist * invDist;
      const float strength = pj.w * invDistCubed;

      ax += strength * dx;
      ay += strength * dy;
      az += strength * dz;
    }
    __syncthreads();
  }

  if (i < n) {
    acc[i] = make_float4(ax, ay, az, 0.0f);
  }
}

// Semi-implicit Euler: velocity updates from the acceleration computed above,
// then position updates from the new velocity. Output-centric, no atomics.
// Identical to Stage 1: integration is O(N), not the part tiling targets.
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
  std::string csvPath = "results/stage2_benchmark.csv";
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
        computeForcesTiledKernel<<<blocks, kBlockSize>>>(dPos, dAcc, n, config.epsilonSquared);
        integrateKernel<<<blocks, kBlockSize>>>(dPos, dVel, dAcc, n, config.dt);
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      const double start = nowMs();
      for (int step = 0; step < config.steps; ++step) {
        computeForcesTiledKernel<<<blocks, kBlockSize>>>(dPos, dAcc, n, config.epsilonSquared);
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
