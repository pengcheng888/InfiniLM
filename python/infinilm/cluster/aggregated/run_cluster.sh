#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/data-aisoft/mechdancer/models/Qwen3-0.6B}"
NAMESPACE="${NAMESPACE:-infinilm-agg}"
RUN_DIR="${RUN_DIR:-/tmp/infinilm-agg}"
FILE_KV="${RUN_DIR}/discovery"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-0.6B}"
# ROUTER_MODE="${ROUTER_MODE:-round-robin}"
ROUTER_MODE="${ROUTER_MODE:-kv-aware}"

DISCOVERY_BACKEND="${DISCOVERY_BACKEND:-file}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-127.0.0.1:2379}"
ETCD_PREFIX="${ETCD_PREFIX:-/infinilm}"
CACHE_INDEX_DIR="${CACHE_INDEX_DIR:-}"

export INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS="${INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS:-${HEALTH_INTERVAL:-1}}"
export INFINILM_CLUSTER_DRAIN_GRACE_SECONDS="${INFINILM_CLUSTER_DRAIN_GRACE_SECONDS:-${DRAIN_GRACE_SECONDS:-1}}"
GPUS=(${GPUS:-0 1 2})


FRONTEND_PORT="${FRONTEND_PORT:-18000}"
WORKER_HTTP_BASE_PORT="${WORKER_HTTP_BASE_PORT:-18081}"

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
  if [[ "$ROUTER_MODE" == kv-aware && -z "$CACHE_INDEX_DIR" ]]; then
    echo "run_cluster: etcd + kv-aware requires CACHE_INDEX_DIR on shared storage" >&2
    exit 1
  fi
  [[ -z "$CACHE_INDEX_DIR" ]] || discovery_args+=(--cache-index-dir "$CACHE_INDEX_DIR")
else
  mkdir -p "$FILE_KV"
fi
pids=()

cleanup() {
  local status=$?
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
    fi
  done
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || true
    fi
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

python -m infinilm.cluster.aggregated.frontend \
  --namespace "$NAMESPACE" \
  "${discovery_args[@]}" \
  --router-mode "$ROUTER_MODE" \
  --router-min-initial-workers "${#GPUS[@]}" \
  --http-host 127.0.0.1 \
  --http-port "$FRONTEND_PORT" &
pids+=("$!")

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  http_port=$((WORKER_HTTP_BASE_PORT + i + 10))

  CUDA_VISIBLE_DEVICES="$gpu" \
  python -m infinilm.cluster.worker \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --namespace "$NAMESPACE" \
    "${discovery_args[@]}" \
    --advertise-host 127.0.0.1 \
    --host 127.0.0.1 \
    --port "$http_port" \
    --device nvidia \
    --tp 1 \
    --block-size 256 \
    --max-new-tokens 2644 \
    --num-blocks 500 \
    --max-batch-size 32 \
    --log-level INFO \
    --enable-paged-attn \
    --attn flash-attn \
    --enable-graph &
  pids+=("$!")
done

echo "frontend: http://127.0.0.1:${FRONTEND_PORT}/v1"
echo "monitoring: http://127.0.0.1:${FRONTEND_PORT}/monitoring"
echo "discovery: $FILE_KV"

# A worker crash is a routing-domain failure; frontend remains available with
# the remaining replicas. When frontend exits, cleanup reclaims every worker.
wait "${pids[0]}"
