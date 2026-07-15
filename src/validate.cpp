/* Compares two --dump-final position CSVs (one per stage) body-by-body and
reports the max/mean positional deviation. Stages agree on initial
conditions because they share the same seeded RNG in common.h, so body i
in one dump corresponds directly to body i in the other; no matching or
sorting is needed. Some deviation is expected since Stage 0 accumulates
forces in double precision while GPU stages use float, so this reports a
tolerance rather than requiring exact agreement. */
#include "common.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: validate <dump_a.csv> <dump_b.csv> [--tol <value>]\n";
    return 2;
  }

  const std::string pathA = argv[1];
  const std::string pathB = argv[2];
  double tolerance = 1e-2;

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tol" && i + 1 < argc) {
      tolerance = std::stod(argv[++i]);
    }
  }

  std::vector<float4> a;
  std::vector<float4> b;
  if (!readPositionsCsv(pathA, a)) {
    std::cerr << "failed to read " << pathA << '\n';
    return 2;
  }
  if (!readPositionsCsv(pathB, b)) {
    std::cerr << "failed to read " << pathB << '\n';
    return 2;
  }

  if (a.size() != b.size()) {
    std::cerr << "body count mismatch: " << a.size() << " vs " << b.size() << '\n';
    return 2;
  }

  double maxDist = 0.0;
  double sumDist = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double dx = static_cast<double>(a[i].x) - static_cast<double>(b[i].x);
    const double dy = static_cast<double>(a[i].y) - static_cast<double>(b[i].y);
    const double dz = static_cast<double>(a[i].z) - static_cast<double>(b[i].z);
    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    maxDist = std::max(maxDist, dist);
    sumDist += dist;
  }
  const double meanDist = sumDist / static_cast<double>(a.size());

  const bool pass = maxDist <= tolerance;
  std::cout << "bodies=" << a.size()
            << " max_dist=" << maxDist
            << " mean_dist=" << meanDist
            << " tolerance=" << tolerance
            << " result=" << (pass ? "PASS" : "FAIL") << '\n';

  return pass ? 0 : 1;
}
