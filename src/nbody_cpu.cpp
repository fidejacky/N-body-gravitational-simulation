#include "common.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  SimConfig config = defaultConfig();
  std::string csvPath = "results/stage0_benchmark.csv";
  bool fullSweep = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--n" && i + 1 < argc) {
      config.numBodies = std::stoi(argv[++i]);
    } else if (arg == "--steps" && i + 1 < argc) {
      config.steps = std::stoi(argv[++i]);
    } else if (arg == "--repeats" && i + 1 < argc) {
      config.repeats = std::stoi(argv[++i]);
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
    std::vector<double> runTimes;
    runTimes.reserve(config.repeats);

    for (int run = 0; run < config.repeats; ++run) {
      std::vector<float4> positions(static_cast<size_t>(n));
      std::vector<float4> velocities(static_cast<size_t>(n));
      std::vector<float4> accelerations(static_cast<size_t>(n), make_float4(0.0f, 0.0f, 0.0f, 0.0f));

      initializeBodies(positions, velocities, n, config.seed + run, config.useCollidingClusters);

      const double start = nowMs();
      for (int step = 0; step < config.steps; ++step) {
        for (int i = 0; i < n; ++i) {
          double ax = 0.0;
          double ay = 0.0;
          double az = 0.0;

          for (int j = 0; j < n; ++j) {
            if (i == j) {
              continue;
            }

            const double dx = static_cast<double>(positions[j].x) - static_cast<double>(positions[i].x);
            const double dy = static_cast<double>(positions[j].y) - static_cast<double>(positions[i].y);
            const double dz = static_cast<double>(positions[j].z) - static_cast<double>(positions[i].z);
            const double distSq = dx * dx + dy * dy + dz * dz + static_cast<double>(config.epsilonSquared);
            const double invDist = 1.0 / std::sqrt(distSq);
            const double invDistCubed = invDist * invDist * invDist;
            const double strength = static_cast<double>(positions[j].w) * invDistCubed;

            ax += strength * dx;
            ay += strength * dy;
            az += strength * dz;
          }

          accelerations[i] = make_float4(static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(az), 0.0f);
        }

        for (int i = 0; i < n; ++i) {
          velocities[i].x += static_cast<float>(accelerations[i].x * config.dt);
          velocities[i].y += static_cast<float>(accelerations[i].y * config.dt);
          velocities[i].z += static_cast<float>(accelerations[i].z * config.dt);

          positions[i].x += velocities[i].x * config.dt;
          positions[i].y += velocities[i].y * config.dt;
          positions[i].z += velocities[i].z * config.dt;
        }
      }

      runTimes.push_back(nowMs() - start);
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
