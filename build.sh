#!/usr/bin/env bash
# One-command build for Linux/macOS: configures and builds all stages via CMake + Ninja.
set -euo pipefail

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
