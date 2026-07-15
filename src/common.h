#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <vector_functions.h>
#else
struct float4 {
  float x;
  float y;
  float z;
  float w;
};

inline float4 make_float4(float x, float y, float z, float w) {
  return float4{x, y, z, w};
}
#endif

inline float4 operator+(const float4& a, const float4& b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline float4 operator-(const float4& a, const float4& b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline float4 operator*(const float4& a, float s) {
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline float4 operator*(float s, const float4& a) {
  return a * s;
}

inline float4 operator/(const float4& a, float s) {
  return make_float4(a.x / s, a.y / s, a.z / s, a.w / s);
}

inline float4& operator+=(float4& a, const float4& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

inline float4& operator-=(float4& a, const float4& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

struct SimConfig {
  int numBodies{4096};
  int steps{10};
  int warmup{1};
  int repeats{3};
  float dt{0.01f};
  float epsilonSquared{0.01f};
  int seed{42};
  bool useCollidingClusters{false};
};

inline SimConfig defaultConfig() {
  return SimConfig{};
}

inline void initializeBodies(std::vector<float4>& positions,
                             std::vector<float4>& velocities,
                             int n,
                             int seed,
                             bool useCollidingClusters = false) {
  if (positions.size() != static_cast<size_t>(n) || velocities.size() != static_cast<size_t>(n)) {
    throw std::invalid_argument("position and velocity buffers must match the requested size");
  }

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  constexpr float kPi = 3.14159265358979323846f;

  if (useCollidingClusters) {
    const int half = n / 2;
    for (int i = 0; i < n; ++i) {
      const float sign = (i < half) ? -1.0f : 1.0f;
      const float radius = 0.15f + 0.05f * std::abs(dist(rng));
      const float theta = dist(rng) * kPi;
      const float cx = sign * 1.0f;
      const float px = cx + radius * std::cos(theta);
      const float py = radius * std::sin(theta) * 0.5f;
      const float pz = 0.05f * dist(rng);
      const float vx = -sign * 0.3f + 0.01f * dist(rng);
      const float vy = 0.02f * dist(rng);
      const float vz = 0.01f * dist(rng);
      positions[i] = make_float4(px, py, pz, 1.0f);
      velocities[i] = make_float4(vx, vy, vz, 0.0f);
    }
  } else {
    for (int i = 0; i < n; ++i) {
      const float radius = 0.2f + 0.8f * std::sqrt(std::abs(dist(rng)) + 1e-6f);
      const float theta = dist(rng) * 2.0f * kPi;
      const float x = radius * std::cos(theta);
      const float y = radius * std::sin(theta);
      const float z = 0.01f * dist(rng);
      const float speed = 0.35f / std::sqrt(radius + 0.2f);
      const float vx = -speed * std::sin(theta);
      const float vy = speed * std::cos(theta);
      const float vz = 0.01f * dist(rng);
      positions[i] = make_float4(x, y, z, 1.0f);
      velocities[i] = make_float4(vx, vy, vz, 0.0f);
    }
  }
}

inline double nowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

inline bool writeCsvHeader(const std::string& path, const std::vector<std::string>& header) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    return false;
  }
  for (size_t i = 0; i < header.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << header[i];
  }
  out << '\n';
  return true;
}

inline bool appendCsvRow(const std::string& path, const std::vector<std::string>& values) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream out(path, std::ios::app);
  if (!out) {
    return false;
  }
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << values[i];
  }
  out << '\n';
  return true;
}

inline bool writePositionsCsv(const std::string& path, const std::vector<float4>& positions) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    return false;
  }
  out << "x,y,z,mass\n";
  out << std::setprecision(9);
  for (const auto& p : positions) {
    out << p.x << ',' << p.y << ',' << p.z << ',' << p.w << '\n';
  }
  return true;
}

inline bool readPositionsCsv(const std::string& path, std::vector<float4>& positions) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }

  std::string line;
  std::getline(in, line);  // header

  positions.clear();
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream lineStream(line);
    std::string token;
    float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4 && std::getline(lineStream, token, ','); ++i) {
      values[i] = std::stof(token);
    }
    positions.push_back(make_float4(values[0], values[1], values[2], values[3]));
  }
  return true;
}
