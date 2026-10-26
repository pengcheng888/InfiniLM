#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

MODE="${1:-${MODE:-}}"
MODEL="${MODEL:-/data-aisoft/mechdancer/models/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-0.6B}"
HOST="${HOST:-127.0.0.1}"
NAMESPACE="${NAMESPACE:-infinilm-pd}"
RUN_DIR="${RUN_DIR:-/tmp/infinilm-pd}"
FILE_KV="${FILE_KV:-$RUN_DIR/discovery}"
DISCOVERY_BACKEND="${DISCOVERY_BACKEND:-file}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-127.0.0.1:2379}"
ETCD_PREFIX="${ETCD_PREFIX:-/infinilm}"
export INFINILM_CLUSTER_DRAIN_GRACE_SECONDS="${INFINILM_CLUSTER_DRAIN_GRACE_SECONDS:-${DRAIN_GRACE_SECONDS:-3}}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
MOONCAKE_WORKERS="${MOONCAKE_WORKERS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ENABLE_GRAPH="${ENABLE_GRAPH:-1}"
EXTRA_ARGS=(${EXTRA_ARGS:-})

case "$MODE" in
  prefill)
    GPU="${GPU:-${PREFILL_GPU:-4}}"
    PORT="${PORT:-${PREFILL_PORT:-18521}}"
    BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-${PREFILL_BOOTSTRAP_PORT:-18528}}"
    ;;
  decode)
    GPU="${GPU:-${DECODE_GPU:-5}}"
    PORT="${PORT:-${DECODE_PORT:-18531}}"
    BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-${DECODE_BOOTSTRAP_PORT:-18528}}"
    ;;
  *)
    echo "usage: $0 {prefill|decode}   # role may also be set with MODE" >&2
    exit 2
    ;;
esac

if [[ -n "${ENGINE_ID:-}" ]]; then
  :
elif [[ "$PORT" == 0 ]]; then
  # port=0 时不能用固定端口号构造 engine ID；加入当前脚本 PID 保证多实例唯一。
  ENGINE_ID="${NAMESPACE}-${MODE}-auto-$$"
else
  ENGINE_ID="${NAMESPACE}-${MODE}-${PORT}"
fi

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "run_pd_worker: Python interpreter not found: $PYTHON" >&2
  exit 1
}
[[ -d "$MODEL" ]] || {
  echo "run_pd_worker: model directory does not exist: $MODEL" >&2
  exit 1
}
[[ "$MOONCAKE_PROTOCOL" == tcp || "$MOONCAKE_PROTOCOL" == rdma ]] || {
  echo "run_pd_worker: MOONCAKE_PROTOCOL must be tcp or rdma" >&2
  exit 1
}
[[ "$HOST" != "0.0.0.0" ]] || {
  echo "run_pd_worker: HOST=0.0.0.0 cannot be advertised; use a reachable IP" >&2
  exit 1
}

if ! "$PYTHON" -c 'import infinilm' >/dev/null 2>&1; then
  export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
fi
"$PYTHON" -c 'import mooncake' >/dev/null 2>&1 || {
  echo "run_pd_worker: the mooncake Python package is required" >&2
  exit 1
}

cuda_runtime_dir="${CUDA12_RUNTIME_DIR:-}"
if [[ -z "$cuda_runtime_dir" ]]; then
  cuda_runtime_dir="$(
    "$PYTHON" - <<'PY' 2>/dev/null || true
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("nvidia.cuda_runtime")
if spec is not None and spec.origin is not None:
    print(Path(spec.origin).parent / "lib")
PY
  )"
fi
if [[ -n "$cuda_runtime_dir" && -d "$cuda_runtime_dir" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/root/.infini/lib}:$cuda_runtime_dir"
fi

graph_args=()
if [[ "$ENABLE_GRAPH" == 1 || "$ENABLE_GRAPH" == true || "$ENABLE_GRAPH" == yes ]]; then
  graph_args=(--enable-graph)
fi

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

kv_transfer_config="$(
  printf '{"kv_connector":"MooncakeConnector","engine_id":"%s","kv_connector_extra_config":{"mooncake_protocol":"%s","num_workers":%s}}' \
    "$ENGINE_ID" "$MOONCAKE_PROTOCOL" "$MOONCAKE_WORKERS"
)"

http_display="${HOST}:auto"
bootstrap_display="${HOST}:auto"
[[ "$PORT" == 0 ]] || http_display="${HOST}:${PORT}"
[[ "$BOOTSTRAP_PORT" == 0 ]] || bootstrap_display="${HOST}:${BOOTSTRAP_PORT}"
echo "[$MODE] gpu=$GPU http=${http_display} bootstrap=${bootstrap_display} engine_id=$ENGINE_ID" >&2
echo "[$MODE] discovery=$DISCOVERY_BACKEND namespace=$NAMESPACE model=$SERVED_MODEL_NAME" >&2
exec env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  INFINILM_HOST_IP="$HOST" \
  INFINILM_MOONCAKE_BOOTSTRAP_HOST="$HOST" \
  INFINILM_MOONCAKE_BOOTSTRAP_PORT="$BOOTSTRAP_PORT" \
  "$PYTHON" -m infinilm.cluster.worker \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --namespace "$NAMESPACE" \
    "${discovery_args[@]}" \
    --advertise-host "$HOST" \
    --disaggregation-mode "$MODE" \
    --mooncake-bootstrap-host "$HOST" \
    --mooncake-bootstrap-port "$BOOTSTRAP_PORT" \
    --host "$HOST" \
    --port "$PORT" \
    --device nvidia \
    --tp 1 \
    --block-size 256 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --log-level "$LOG_LEVEL" \
    --enable-paged-attn \
    --attn flash-attn \
    "${graph_args[@]}" \
    --kv-transfer-config "$kv_transfer_config" \
    "${EXTRA_ARGS[@]}"
