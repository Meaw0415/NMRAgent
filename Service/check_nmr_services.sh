#!/usr/bin/env bash
set -euo pipefail

RETRIEVAL_URL="${NMR_RETRIEVAL_SERVICE_URL:-http://127.0.0.1:${NMR_RETRIEVAL_SERVICE_PORT:-8011}}"
DENOVO_URL="${NMR_DENOVO_SERVICE_URL:-http://127.0.0.1:${NMR_DENOVO_SERVICE_PORT:-8012}}"

echo "Retrieval health: $RETRIEVAL_URL/health"
curl -sS "$RETRIEVAL_URL/health" || true
echo
echo "Denovo health: $DENOVO_URL/health"
curl -sS "$DENOVO_URL/health" || true
echo
