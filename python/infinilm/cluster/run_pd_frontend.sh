#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

HOST="${HOST:-127.0.0.1}"
NAMESPACE="${NAMESPACE:-infinilm-pd}"
RUN_DIR="${RUN_DIR:-/tmp/infinilm-pd}"
FILE_KV="${FILE_KV:-$RUN_DIR/discovery}"
ROUTER_MODE="${ROUTER_MODE:-round-robin}"
PORT="${PD_FRONTEND_PORT:-18520}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1000}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-1}"

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "run_pd_frontend: Python interpreter not found: $PYTHON" >&2
  exit 1
}
if ! "$PYTHON" -c 'import infinilm' >/dev/null 2>&1; then
  export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
fi

mkdir -p "$FILE_KV"

echo "PD API:       http://${HOST}:${PORT}/v1" >&2
echo "Monitoring:   http://${HOST}:${PORT}/monitoring" >&2
echo "Discovery:    $FILE_KV" >&2
exec "$PYTHON" -m infinilm.cluster.pd_frontend \
  --namespace "$NAMESPACE" \
  --discovery-backend file \
  --file-kv "$FILE_KV" \
  --request-plane http \
  --router-mode "$ROUTER_MODE" \
  --http-host "$HOST" \
  --http-port "$PORT" \
  --connect-timeout "$CONNECT_TIMEOUT" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --health-interval "$HEALTH_INTERVAL"
