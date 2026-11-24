#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${NMR_PYTHON:-python}" "$repo_dir/scripts/run_benchmark.py" \
  --benchmark nmrshiftdb \
  --input "${NMRSHIFTDB_INPUT:-$repo_dir/artifacts/benchmarks/nmrshiftdb_test.json}" \
  --output-dir "$repo_dir/results/nmrshiftdb" "$@"
