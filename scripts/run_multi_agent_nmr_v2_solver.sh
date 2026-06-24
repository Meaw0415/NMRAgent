#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${NMR_SOLVER_PYTHON:-python}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" scripts/run_multi_agent_nmr.py "$@"
