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

/* One thread per body, output-centric: each thread accumulates only its own
body's acceleration and writes it once, so no atomics are needed. The i==j
term is left in the loop rather than branched around: r_j - r_i is the zero
vector when j==i, so it contributes nothing to the sum regardless of the
softened denominator, and skipping the branch avoids per-thread divergence. */
__global__ void computeForcesKernel(const float4* pos, float4* acc, int n, float epsSq) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const float4 pi = pos[i];
  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;

  for (int j = 0; j < n; ++j) {
    const float4 pj = pos[j];
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

  acc[i] = make_float4(ax, ay, az, 0.0f);
}

// Semi-implicit Euler: velocity updates from the acceleration computed above,
// then position updates from the new velocity. Output-centric, no atomics.
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
  std::string csvPath = "results/stage1_benchmark.csv";
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
        computeForcesKernel<<<blocks, kBlockSize>>>(dPos, dAcc, n, config.epsilonSquared);
        integrateKernel<<<blocks, kBlockSize>>>(dPos, dVel, dAcc, n, config.dt);
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      const double start = nowMs();
      for (int step = 0; step < config.steps; ++step) {
        computeForcesKernel<<<blocks, kBlockSize>>>(dPos, dAcc, n, config.epsilonSquared);
        integrateKernel<<<blocks, kBlockSize>>>(dPos, dVel, dAcc, n, config.dt);
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      runTimes.push_back(nowMs() - start);
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
