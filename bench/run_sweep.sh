#!/usr/bin/env bash
# Stage 4 block-size sweep. Runs nbody_tuned across block sizes 64/128/256/
# 512/1024 and the full N sweep for each, then merges into one CSV. Run from
# the repository root after building (./build.ps1 or ./build.sh).
#
# nbody_tuned truncates its --csv path at the start of every invocation, so
# each block size is written to its own temp file first and merged afterward
# rather than appended directly, which would silently wipe prior block sizes.
set -euo pipefail

BLOCK_SIZES=(64 128 256 512 1024)
OUT_CSV="results/stage4_benchmark.csv"
TMP_DIR="results/stage4_tmp"

mkdir -p "$TMP_DIR"

for bs in "${BLOCK_SIZES[@]}"; do
  echo "=== block-size=$bs ==="
  ./results/nbody_tuned.exe --sweep --steps 100 --repeats 3 --warmup 1 \
    --block-size "$bs" --csv "$TMP_DIR/bs_$bs.csv"
done

head -n 1 "$TMP_DIR/bs_${BLOCK_SIZES[0]}.csv" > "$OUT_CSV"
for bs in "${BLOCK_SIZES[@]}"; do
  tail -n +2 "$TMP_DIR/bs_$bs.csv" >> "$OUT_CSV"
done

rm -rf "$TMP_DIR"
echo "Wrote $OUT_CSV"
