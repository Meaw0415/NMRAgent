#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${NMR_PYTHON:-python}" "$repo_dir/scripts/run_benchmark.py" \
  --benchmark exp450 \
  --input "${EXP450_INPUT:-$repo_dir/artifacts/benchmarks/exp450.json}" \
  --output-dir "$repo_dir/results/exp450" "$@"
