#!/usr/bin/env bash
set -euo pipefail

# Start the local NMRAgent chat stack in detached tmux sessions.
# Services:
#   retrieval: http://127.0.0.1:8011
#   denovo:    http://127.0.0.1:8012
#   chat UI:   http://127.0.0.1:7860

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${NMR_STACK_LOG_DIR:-$REPO_ROOT/logs/services}"
PYTHON_BIN="${NMR_SERVICE_PYTHON:-python}"
CHAT_HOST="${NMR_CHAT_HOST:-0.0.0.0}"
CHAT_PORT="${NMR_CHAT_PORT:-7860}"
RETRIEVAL_PORT="${NMR_RETRIEVAL_SERVICE_PORT:-8011}"
DENOVO_PORT="${NMR_DENOVO_SERVICE_PORT:-8012}"
RETRIEVAL_SESSION="${NMR_RETRIEVAL_TMUX_SESSION:-nmr_retrieval_service}"
DENOVO_SESSION="${NMR_DENOVO_TMUX_SESSION:-nmr_denovo_service}"
CHAT_SESSION="${NMR_CHAT_TMUX_SESSION:-nmragent_chat_langgraph}"
RESTART="${NMR_STACK_RESTART:-1}"
mkdir -p "$LOG_DIR"

start_session() {
  local session="$1"
  local command="$2"
  if tmux has-session -t "$session" 2>/dev/null; then
    if [[ "$RESTART" == "1" || "$RESTART" == "true" || "$RESTART" == "yes" ]]; then
      tmux kill-session -t "$session"
    else
      echo "tmux session already exists: $session"
      return 0
    fi
  fi
  tmux new-session -d -s "$session" "$command"
  echo "started tmux session: $session"
}

RETRIEVAL_URL="http://127.0.0.1:${RETRIEVAL_PORT}"
DENOVO_URL="http://127.0.0.1:${DENOVO_PORT}"

start_session "$RETRIEVAL_SESSION" \
  "cd '$REPO_ROOT' && NMR_RETRIEVAL_PRELOAD='${NMR_RETRIEVAL_PRELOAD:-0}' NMR_RETRIEVAL_SERVICE_PORT='$RETRIEVAL_PORT' NMR_RETRIEVAL_PYTHON='$PYTHON_BIN' bash Service/start_retrieval_service.sh 2>&1 | tee '$LOG_DIR/retrieval_${RETRIEVAL_PORT}.log'"

start_session "$DENOVO_SESSION" \
  "cd '$REPO_ROOT' && NMR_DENOVO_SERVICE_PORT='$DENOVO_PORT' NMR_DENOVO_PYTHON='$PYTHON_BIN' bash Service/start_denovo_service.sh 2>&1 | tee '$LOG_DIR/denovo_${DENOVO_PORT}.log'"

start_session "$CHAT_SESSION" \
  "cd '$REPO_ROOT' && NMR_RETRIEVAL_SERVICE_URL='$RETRIEVAL_URL' NMR_DENOVO_SERVICE_URL='$DENOVO_URL' '$PYTHON_BIN' scripts/chat_agent_web.py --host '$CHAT_HOST' --port '$CHAT_PORT' 2>&1 | tee '$REPO_ROOT/logs/chat_agent_web_${CHAT_PORT}.log'"

cat <<MSG

NMRAgent chat stack started.
  Retrieval: $RETRIEVAL_URL/health
  Denovo:    $DENOVO_URL/health
  Chat UI:   http://127.0.0.1:$CHAT_PORT

Logs:
  $LOG_DIR/retrieval_${RETRIEVAL_PORT}.log
  $LOG_DIR/denovo_${DENOVO_PORT}.log
  $REPO_ROOT/logs/chat_agent_web_${CHAT_PORT}.log

Health check:
  NMR_DENOVO_SERVICE_PORT=$DENOVO_PORT bash Service/check_nmr_services.sh
MSG
