// Stage 5: the same simulation expressed through Thrust instead of a
// hand-written kernel, as a library-based performance ceiling. Per
// CLAUDE.md/project_plan.md, Thrust is used here only for this comparison,
// not as the main implementation. No manual __global__ kernel, no shared-
// memory tiling, no launch configuration tuning: thrust::for_each picks all
// of that for us. The math is the plain, unoptimized all-pairs formula
// (matching Stage 1, not Stage 3's rsqrtf/fmaf), since this is meant to be
// the honest "if I just reached for the library instead of hand-tuning"
// baseline that the hand-tuned stages get measured against.
#include "common.h"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

struct ForceFunctor {
  const float4* pos;
  float4* acc;
  int n;
  float epsSq;

  __device__ void operator()(int i) const {
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
};

struct IntegrateFunctor {
  float dt;

  template <typename Tuple>
  __device__ void operator()(Tuple t) const {
    float4& pos = thrust::get<0>(t);
    float4& vel = thrust::get<1>(t);
    const float4& acc = thrust::get<2>(t);

    vel.x += acc.x * dt;
    vel.y += acc.y * dt;
    vel.z += acc.z * dt;

    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
  }
};

int main(int argc, char** argv) {
  SimConfig config = defaultConfig();
  std::string csvPath = "results/stage5_benchmark.csv";
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
    thrust::device_vector<float4> dPos(n);
    thrust::device_vector<float4> dVel(n);
    thrust::device_vector<float4> dAcc(n);

    std::vector<double> runTimes;
    runTimes.reserve(config.repeats);

    for (int run = 0; run < config.repeats; ++run) {
      std::vector<float4> hPos(static_cast<size_t>(n));
      std::vector<float4> hVel(static_cast<size_t>(n));
      initializeBodies(hPos, hVel, n, config.seed + run, config.useCollidingClusters);

      thrust::copy(hPos.begin(), hPos.end(), dPos.begin());
      thrust::copy(hVel.begin(), hVel.end(), dVel.begin());

      const float4* posPtr = thrust::raw_pointer_cast(dPos.data());
      float4* accPtr = thrust::raw_pointer_cast(dAcc.data());
      const ForceFunctor forceFunctor{posPtr, accPtr, n, config.epsilonSquared};
      const IntegrateFunctor integrateFunctor{config.dt};

      const auto stepOnce = [&]() {
        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), forceFunctor);
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(dPos.begin(), dVel.begin(), dAcc.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(dPos.end(), dVel.end(), dAcc.end())),
            integrateFunctor);
      };

      for (int step = 0; step < config.warmup; ++step) {
        stepOnce();
      }
      cudaDeviceSynchronize();

      const double start = nowMs();
      for (int step = 0; step < config.steps; ++step) {
        stepOnce();
      }
      cudaDeviceSynchronize();
      runTimes.push_back(nowMs() - start);

      if (run == 0 && !dumpFinalPath.empty()) {
        std::vector<float4> finalPos(static_cast<size_t>(n));
        thrust::copy(dPos.begin(), dPos.end(), finalPos.begin());
        writePositionsCsv(dumpFinalPath, finalPos);
      }
    }

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
