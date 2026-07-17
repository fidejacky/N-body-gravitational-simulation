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

template <int BLOCK_SIZE>
__global__ void computeForcesTiledFastKernel(const float4* pos, float4* acc, int n, float epsSq) {
  // Shared-memory tiling
  __shared__ float4 tile[BLOCK_SIZE];

  // One thread per body
  const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const float4 pi = (i < n) ? pos[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;

  const int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int t = 0; t < numTiles; ++t) {
    const int j = t * BLOCK_SIZE + threadIdx.x;
    tile[threadIdx.x] = (j < n) ? pos[j] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    __syncthreads();

// Loop unrolling
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      const float4 pj = tile[k];
      const float dx = pj.x - pi.x;
      const float dy = pj.y - pi.y;
      const float dz = pj.z - pi.z;
      const float distSq = dx * dx + dy * dy + dz * dz + epsSq;
      // Fast-math intrinsics
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

template <int BLOCK_SIZE>
__global__ void integrateKernel(float4* pos, float4* vel, const float4* acc, int n, float dt) {
  const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
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

template <int BLOCK_SIZE>
void runSteps(float4* dPos, float4* dVel, float4* dAcc, int n, float epsSq, float dt, int numSteps) {
  const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int step = 0; step < numSteps; ++step) {
    computeForcesTiledFastKernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(dPos, dAcc, n, epsSq);
    integrateKernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(dPos, dVel, dAcc, n, dt);
  }
}

// Runtime block-size tuning
void runStepsDispatch(int blockSize, float4* dPos, float4* dVel, float4* dAcc, int n, float epsSq, float dt, int numSteps) {
  switch (blockSize) {
    case 64:
      runSteps<64>(dPos, dVel, dAcc, n, epsSq, dt, numSteps);
      break;
    case 128:
      runSteps<128>(dPos, dVel, dAcc, n, epsSq, dt, numSteps);
      break;
    case 256:
      runSteps<256>(dPos, dVel, dAcc, n, epsSq, dt, numSteps);
      break;
    case 512:
      runSteps<512>(dPos, dVel, dAcc, n, epsSq, dt, numSteps);
      break;
    case 1024:
      runSteps<1024>(dPos, dVel, dAcc, n, epsSq, dt, numSteps);
      break;
    default:
      std::cerr << "unsupported --block-size " << blockSize << " (use 64, 128, 256, 512, or 1024)\n";
      std::exit(1);
  }
}

int main(int argc, char** argv) {
  SimConfig config = defaultConfig();
  std::string csvPath = "results/stage4_benchmark.csv";
  std::string dumpFinalPath;
  int blockSize = 256;
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
    } else if (arg == "--block-size" && i + 1 < argc) {
      blockSize = std::stoi(argv[++i]);
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

  const std::vector<std::string> header = {"n", "block_size", "steps", "repeats", "avg_ms_per_step", "avg_total_ms"};
  writeCsvHeader(csvPath, header);

  for (int n : sizes) {
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

      runStepsDispatch(blockSize, dPos, dVel, dAcc, n, config.epsilonSquared, config.dt, config.warmup);
      CUDA_CHECK(cudaDeviceSynchronize());

      const double start = nowMs();
      runStepsDispatch(blockSize, dPos, dVel, dAcc, n, config.epsilonSquared, config.dt, config.steps);
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
        std::to_string(blockSize),
        std::to_string(config.steps),
        std::to_string(config.repeats),
        std::to_string(avgPerStepMs),
        std::to_string(avgTotalMs)
    });

    std::cout << "n=" << n
              << " block_size=" << blockSize
              << " avg_ms_per_step=" << avgPerStepMs
              << " avg_total_ms=" << avgTotalMs << '\n';
  }

  return 0;
}
