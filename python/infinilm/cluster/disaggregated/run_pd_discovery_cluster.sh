#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

MODEL="${MODEL:-/data-aisoft/mechdancer/models/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-0.6B}"
HOST="${HOST:-127.0.0.1}"
NAMESPACE="${NAMESPACE:-infinilm-pd}"
RUN_DIR="${RUN_DIR:-/tmp/infinilm-pd}"
FILE_KV="${FILE_KV:-$RUN_DIR/discovery}"
ROUTER_MODE="${ROUTER_MODE:-round-robin}"
DISCOVERY_BACKEND="${DISCOVERY_BACKEND:-file}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-127.0.0.1:2379}"
ETCD_PREFIX="${ETCD_PREFIX:-/infinilm}"
PROXY_PORT="${PD_FRONTEND_PORT:-18520}"
PREFILL_PORT_BASE="${PREFILL_PORT_BASE:-${PREFILL_PORT:-18521}}"
DECODE_PORT_BASE="${DECODE_PORT_BASE:-${DECODE_PORT:-$((${PREFILL_PORT:-18521} + 10))}}"
BOOTSTRAP_PORT_BASE="${BOOTSTRAP_PORT_BASE:-${BOOTSTRAP_PORT:-18528}}"
PREFILL_GPU="${PREFILL_GPU:-4}"
DECODE_GPU="${DECODE_GPU:-5}"
PREFILL_GPUS="${PREFILL_GPUS:-$PREFILL_GPU}"
DECODE_GPUS="${DECODE_GPUS:-$DECODE_GPU}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
MOONCAKE_WORKERS="${MOONCAKE_WORKERS:-4}"
export INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS="${INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS:-${HEALTH_INTERVAL:-1}}"
export INFINILM_CLUSTER_DRAIN_GRACE_SECONDS="${INFINILM_CLUSTER_DRAIN_GRACE_SECONDS:-${DRAIN_GRACE_SECONDS:-3}}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
NUM_BLOCKS="${NUM_BLOCKS:-500}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ENABLE_GRAPH="${ENABLE_GRAPH:-1}"
PLAN_ONLY="${PLAN_ONLY:-0}"
EXTRA_ARGS=(${EXTRA_ARGS:-})

pids=()
prefill_pids=()
decode_pids=()
engine_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

fail() {
  echo "run_pd_discovery_cluster: $*" >&2
  exit 1
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while ((SECONDS < deadline)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || true
      fail "$name exited before becoming ready"
    fi
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  fail "timed out waiting for $name at $url"
}

planned_port() {
  local base="$1"
  local index="$2"
  if (( base == 0 )); then
    echo 0
  else
    echo "$((base + index))"
  fi
}

wait_for_pd_ready() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  local health
  while ((SECONDS < deadline)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || true
      fail "$name exited before becoming ready"
    fi
    health="$(curl -fsS --max-time 2 "$url/health" 2>/dev/null || true)"
    if "$PYTHON" -c '
import json
import sys

try:
    ready = json.loads(sys.argv[1]).get("ready") is True
except Exception:
    ready = False
sys.exit(0 if ready else 1)
' "$health"; then
      return 0
    fi
    sleep 1
  done
  fail "timed out waiting for $name to report ready=true at $url/health"
}

command -v "$PYTHON" >/dev/null 2>&1 || fail "Python interpreter not found: $PYTHON"
[[ -d "$MODEL" ]] || fail "model directory does not exist: $MODEL"
[[ "$DISCOVERY_BACKEND" == file || "$DISCOVERY_BACKEND" == etcd ]] ||
  fail "DISCOVERY_BACKEND must be file or etcd"
[[ "$MOONCAKE_PROTOCOL" == tcp || "$MOONCAKE_PROTOCOL" == rdma ]] ||
  fail "MOONCAKE_PROTOCOL must be tcp or rdma"
[[ "$ROUTER_MODE" == round-robin || "$ROUTER_MODE" == least-loaded ]] ||
  fail "PD ROUTER_MODE must be round-robin or least-loaded"
[[ "$HOST" != "0.0.0.0" ]] ||
  fail "HOST=0.0.0.0 cannot be advertised; use a reachable IP"

if ! "$PYTHON" -c 'import infinilm' >/dev/null 2>&1; then
  export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
fi
"$PYTHON" -c 'import mooncake' >/dev/null 2>&1 ||
  fail "the mooncake Python package is required"
if [[ "$DISCOVERY_BACKEND" == etcd ]]; then
  "$PYTHON" -c 'import etcd3' >/dev/null 2>&1 ||
    fail "etcd discovery requires: python -m pip install -e '.[etcd]'"
fi

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
else
  mkdir -p "$FILE_KV"
  rm -f "$FILE_KV/${NAMESPACE}."*.json
fi

graph_args=()
if [[ "$ENABLE_GRAPH" == 1 || "$ENABLE_GRAPH" == true || "$ENABLE_GRAPH" == yes ]]; then
  graph_args=(--enable-graph)
fi

start_engine() {
  local role="$1"
  local disaggregation_mode="$2"
  local gpu="$3"
  local port="$4"
  local bootstrap_port="$5"
  local engine_id="$6"
  local kv_config
  kv_config="$(
    printf '{"kv_connector":"MooncakeConnector","engine_id":"%s","kv_connector_extra_config":{"mooncake_protocol":"%s","num_workers":%s}}' \
      "$engine_id" "$MOONCAKE_PROTOCOL" "$MOONCAKE_WORKERS"
  )"

  http_display="${HOST}:auto"
  [[ "$port" == 0 ]] || http_display="${HOST}:${port}"
  echo "[$role] gpu=$gpu http=${http_display} engine_id=$engine_id mode=$disaggregation_mode" >&2
  CUDA_VISIBLE_DEVICES="$gpu" \
  "$PYTHON" -m infinilm.cluster.worker \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --namespace "$NAMESPACE" \
    "${discovery_args[@]}" \
    --advertise-host "$HOST" \
    --disaggregation-mode "$disaggregation_mode" \
    --mooncake-bootstrap-host "$HOST" \
    --mooncake-bootstrap-port "$bootstrap_port" \
    --host "$HOST" \
    --port "$port" \
    --device nvidia \
    --tp 1 \
    --block-size 256 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --num-blocks "$NUM_BLOCKS" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --log-level "$LOG_LEVEL" \
    --enable-paged-attn \
    --attn flash-attn \
    "${graph_args[@]}" \
    --kv-transfer-config "$kv_config" \
    "${EXTRA_ARGS[@]}" &
  pids+=("$!")
  engine_pid="$!"
}

frontend_url="http://${HOST}:${PROXY_PORT}"
read -r -a PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
read -r -a DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
(( ${#PREFILL_GPU_ARRAY[@]} > 0 )) || fail "PREFILL_GPUS must not be empty"
(( ${#DECODE_GPU_ARRAY[@]} > 0 )) || fail "DECODE_GPUS must not be empty"

port_checks=("frontend" "$PROXY_PORT")
for i in "${!PREFILL_GPU_ARRAY[@]}"; do
  prefill_http_port="$(planned_port "$PREFILL_PORT_BASE" "$i")"
  prefill_bootstrap_port="$(planned_port "$BOOTSTRAP_PORT_BASE" "$i")"
  [[ "$prefill_http_port" == 0 ]] ||
    port_checks+=("prefill-$i-http" "$prefill_http_port")
  [[ "$prefill_bootstrap_port" == 0 ]] ||
    port_checks+=("prefill-$i-bootstrap" "$prefill_bootstrap_port")
done
for i in "${!DECODE_GPU_ARRAY[@]}"; do
  decode_http_port="$(planned_port "$DECODE_PORT_BASE" "$i")"
  [[ "$decode_http_port" == 0 ]] ||
    port_checks+=("decode-$i-http" "$decode_http_port")
done
"$PYTHON" - "$HOST" "${port_checks[@]}" <<'PY'
import socket
import sys
from collections import Counter

host = sys.argv[1]
checks = list(zip(sys.argv[2::2], sys.argv[3::2]))
duplicates = [port for port, count in Counter(int(port) for _, port in checks).items() if count > 1]
if duplicates:
    raise SystemExit(f"duplicate planned ports: {duplicates}")
for name, port in checks:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, int(port)))
    except OSError as exc:
        raise SystemExit(f"cannot use {name} port {port}: {exc}") from exc
    finally:
        sock.close()
PY
if [[ "$PLAN_ONLY" == 1 || "$PLAN_ONLY" == true || "$PLAN_ONLY" == yes ]]; then
  prefill_plan="auto"
  decode_plan="auto"
  bootstrap_plan="auto"
  [[ "$PREFILL_PORT_BASE" == 0 ]] || prefill_plan="${PREFILL_PORT_BASE}+"
  [[ "$DECODE_PORT_BASE" == 0 ]] || decode_plan="${DECODE_PORT_BASE}+"
  [[ "$BOOTSTRAP_PORT_BASE" == 0 ]] || bootstrap_plan="${BOOTSTRAP_PORT_BASE}+"
  echo "PD plan validated: frontend=${HOST}:${PROXY_PORT} prefill=${PREFILL_GPU_ARRAY[*]}@${prefill_plan} decode=${DECODE_GPU_ARRAY[*]}@${decode_plan} bootstrap=${bootstrap_plan}"
  exit 0
fi

echo "[frontend] http=${HOST}:${PROXY_PORT} router=$ROUTER_MODE"
"$PYTHON" -m infinilm.cluster.disaggregated.frontend \
  --namespace "$NAMESPACE" \
  "${discovery_args[@]}" \
  --router-mode "$ROUTER_MODE" \
  --http-host "$HOST" \
  --http-port "$PROXY_PORT" &
pids+=("$!")
frontend_pid="$!"

for i in "${!PREFILL_GPU_ARRAY[@]}"; do
  gpu="${PREFILL_GPU_ARRAY[$i]}"
  port="$(planned_port "$PREFILL_PORT_BASE" "$i")"
  bootstrap_port="$(planned_port "$BOOTSTRAP_PORT_BASE" "$i")"
  engine_id="${NAMESPACE}-prefill-${i}"
  start_engine "prefill-$i" prefill "$gpu" "$port" "$bootstrap_port" "$engine_id"
  prefill_pids+=("$engine_pid")
done

for i in "${!DECODE_GPU_ARRAY[@]}"; do
  gpu="${DECODE_GPU_ARRAY[$i]}"
  port="$(planned_port "$DECODE_PORT_BASE" "$i")"
  engine_id="${NAMESPACE}-decode-${i}"
  start_engine "decode-$i" decode "$gpu" "$port" "$BOOTSTRAP_PORT_BASE" "$engine_id"
  decode_pids+=("$engine_pid")
done

if (( PREFILL_PORT_BASE != 0 && BOOTSTRAP_PORT_BASE != 0 && DECODE_PORT_BASE != 0 )); then
  for i in "${!prefill_pids[@]}"; do
    port="$(planned_port "$PREFILL_PORT_BASE" "$i")"
    bootstrap_port="$(planned_port "$BOOTSTRAP_PORT_BASE" "$i")"
    wait_for_url "prefill-$i server" "http://${HOST}:${port}/health" "${prefill_pids[$i]}"
    wait_for_url "prefill-$i bootstrap" "http://${HOST}:${bootstrap_port}/query" "${prefill_pids[$i]}"
  done

  for i in "${!decode_pids[@]}"; do
    port="$(planned_port "$DECODE_PORT_BASE" "$i")"
    wait_for_url "decode-$i server" "http://${HOST}:${port}/health" "${decode_pids[$i]}"
  done
fi

wait_for_pd_ready "PD frontend" "$frontend_url" "$frontend_pid"

cat <<EOF
PD API:       $frontend_url/v1
Monitoring:   $frontend_url/monitoring
Discovery:    $DISCOVERY_BACKEND
Transfer:     mooncake/$MOONCAKE_PROTOCOL
Prefill:      ${PREFILL_GPU_ARRAY[*]} @ ${HOST}:$([[ "$PREFILL_PORT_BASE" == 0 ]] && echo auto || echo "${PREFILL_PORT_BASE}+")
Decode:       ${DECODE_GPU_ARRAY[*]} @ ${HOST}:$([[ "$DECODE_PORT_BASE" == 0 ]] && echo auto || echo "${DECODE_PORT_BASE}+")
Bootstrap:    ${HOST}:$([[ "$BOOTSTRAP_PORT_BASE" == 0 ]] && echo auto || echo "${BOOTSTRAP_PORT_BASE}+")
Press Ctrl+C to stop all services.
EOF

wait -n "${pids[@]}"
status=$?
fail "a PD service exited unexpectedly (status=$status)"
