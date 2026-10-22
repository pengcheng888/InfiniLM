#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

HOST="${HOST:-127.0.0.1}"
NAMESPACE="${NAMESPACE:-infinilm-pd}"
RUN_DIR="${RUN_DIR:-/tmp/infinilm-pd}"
FILE_KV="${FILE_KV:-$RUN_DIR/discovery}"
ROUTER_MODE="${ROUTER_MODE:-round-robin}"
PORT="${PD_FRONTEND_PORT:-18520}"
DISCOVERY_BACKEND="${DISCOVERY_BACKEND:-file}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-127.0.0.1:2379}"
ETCD_PREFIX="${ETCD_PREFIX:-/infinilm}"
export INFINILM_CLUSTER_CONNECT_TIMEOUT="${INFINILM_CLUSTER_CONNECT_TIMEOUT:-${CONNECT_TIMEOUT:-3}}"
export INFINILM_CLUSTER_REQUEST_TIMEOUT="${INFINILM_CLUSTER_REQUEST_TIMEOUT:-${REQUEST_TIMEOUT:-1000}}"
export INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS="${INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS:-${HEALTH_INTERVAL:-1}}"

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "run_pd_frontend: Python interpreter not found: $PYTHON" >&2
  exit 1
}
if ! "$PYTHON" -c 'import infinilm' >/dev/null 2>&1; then
  export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
fi

mkdir -p "$FILE_KV"

discovery_args=(
  --discovery-backend "$DISCOVERY_BACKEND"
  --file-kv "$FILE_KV"
)
if [[ "$DISCOVERY_BACKEND" == etcd ]]; then
  discovery_args+=(
    --etcd-endpoints "$ETCD_ENDPOINTS"
    --etcd-prefix "$ETCD_PREFIX"
    --etcd-username "${ETCD_USERNAME:-}"
    --etcd-password-env "${ETCD_PASSWORD_ENV:-ETCD_PASSWORD}"
    --etcd-ca-cert "${ETCD_CA_CERT:-}"
    --etcd-cert-file "${ETCD_CERT_FILE:-}"
    --etcd-key-file "${ETCD_KEY_FILE:-}"
    --etcd-timeout "${ETCD_TIMEOUT:-3}"
  )
fi

echo "PD API:       http://${HOST}:${PORT}/v1" >&2
echo "Monitoring:   http://${HOST}:${PORT}/monitoring" >&2
if [[ "$DISCOVERY_BACKEND" == etcd ]]; then
  echo "Discovery:    etcd://${ETCD_ENDPOINTS}${ETCD_PREFIX}" >&2
else
  echo "Discovery:    file://$FILE_KV" >&2
fi
exec "$PYTHON" -m infinilm.cluster.disaggregated.frontend \
  --namespace "$NAMESPACE" \
  "${discovery_args[@]}" \
  --router-mode "$ROUTER_MODE" \
  --http-host "$HOST" \
  --http-port "$PORT"
