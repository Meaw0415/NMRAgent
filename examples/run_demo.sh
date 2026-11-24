#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${NMR_PYTHON:-python}" "$repo_dir/scripts/run_benchmark.py" \
  --benchmark case_study \
  --input "$repo_dir/examples/case_studies/compound_1/input.json" \
  --output-dir "$repo_dir/results/compound_1" "$@"
