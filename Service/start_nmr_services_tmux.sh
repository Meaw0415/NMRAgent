#!/usr/bin/env bash
set -euo pipefail

# Start persistent NMRAgent model services in a detached tmux session.
# Services:
#   - retrieval: http://127.0.0.1:8011
#   - denovo:    http://127.0.0.1:8012

SESSION="${NMR_SERVICE_TMUX_SESSION:-nmr_services}"
PYTHON_BIN="${NMR_SERVICE_PYTHON:-python}"
RETRIEVAL_PORT="${NMR_RETRIEVAL_SERVICE_PORT:-8011}"
DENOVO_PORT="${NMR_DENOVO_SERVICE_PORT:-8012}"
HOST="${NMR_SERVICE_HOST:-127.0.0.1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${NMR_SERVICE_LOG_DIR:-$REPO_ROOT/logs/services}"
mkdir -p "$LOG_DIR"

export NMR_RETRIEVAL_SERVICE_HOST="${NMR_RETRIEVAL_SERVICE_HOST:-$HOST}"
export NMR_DENOVO_SERVICE_HOST="${NMR_DENOVO_SERVICE_HOST:-$HOST}"
export NMR_RETRIEVAL_SERVICE_PORT="$RETRIEVAL_PORT"
export NMR_DENOVO_SERVICE_PORT="$DENOVO_PORT"
export NMR_RETRIEVAL_SERVICE_URL="http://${NMR_RETRIEVAL_SERVICE_HOST}:${NMR_RETRIEVAL_SERVICE_PORT}"
export NMR_DENOVO_SERVICE_URL="http://${NMR_DENOVO_SERVICE_HOST}:${NMR_DENOVO_SERVICE_PORT}"
export NMR_RETRIEVAL_PYTHON="${NMR_RETRIEVAL_PYTHON:-$PYTHON_BIN}"
export NMR_DENOVO_PYTHON="${NMR_DENOVO_PYTHON:-$PYTHON_BIN}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n retrieval "cd '$REPO_ROOT' && NMR_RETRIEVAL_PYTHON='$NMR_RETRIEVAL_PYTHON' bash Service/start_retrieval_service.sh 2>&1 | tee '$LOG_DIR/retrieval.log'"
tmux new-window -t "$SESSION" -n denovo "cd '$REPO_ROOT' && NMR_DENOVO_PYTHON='$NMR_DENOVO_PYTHON' bash Service/start_denovo_service.sh 2>&1 | tee '$LOG_DIR/denovo.log'"

echo "Started tmux session: $SESSION"
echo "Retrieval: $NMR_RETRIEVAL_SERVICE_URL"
echo "Denovo:    $NMR_DENOVO_SERVICE_URL"
echo "Logs:      $LOG_DIR"
echo "Attach:    tmux attach -t $SESSION"
echo "Health:    bash Service/check_nmr_services.sh"
