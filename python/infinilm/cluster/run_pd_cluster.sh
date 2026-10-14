#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${MODEL:-/data-aisoft/mechdancer/models/Qwen3-0.6B}"
HOST="${HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-18500}"
PREFILL_PORT="${PREFILL_PORT:-18501}"
DECODE_PORT="${DECODE_PORT:-18502}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-18508}"
PREFILL_GPU="${PREFILL_GPU:-4}"
DECODE_GPU="${DECODE_GPU:-5}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
MOONCAKE_WORKERS="${MOONCAKE_WORKERS:-4}"
ENGINE_ID_PREFIX="${ENGINE_ID_PREFIX:-pd}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
NUM_BLOCKS="${NUM_BLOCKS:-500}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ENABLE_GRAPH="${ENABLE_GRAPH:-1}"
EXTRA_ARGS=(${EXTRA_ARGS:-})

pids=()

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
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
  echo "run_pd_cluster: $*" >&2
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

command -v "$PYTHON" >/dev/null 2>&1 || fail "Python interpreter not found: $PYTHON"
[[ -d "$MODEL" ]] || fail "model directory does not exist: $MODEL"
[[ "$MOONCAKE_PROTOCOL" == tcp || "$MOONCAKE_PROTOCOL" == rdma ]] ||
  fail "MOONCAKE_PROTOCOL must be tcp or rdma"
[[ "$MOONCAKE_WORKERS" =~ ^[0-9]+$ ]] || fail "MOONCAKE_WORKERS must be an integer"
[[ "$HOST" != "0.0.0.0" ]] ||
  fail "HOST=0.0.0.0 cannot be advertised to Mooncake; use a reachable IP"

if ! "$PYTHON" -c 'import infinilm' >/dev/null 2>&1; then
  export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
fi
"$PYTHON" -c 'import mooncake' >/dev/null 2>&1 ||
  fail "the mooncake Python package is required"

# Mooncake 0.3.x links against the CUDA 12 runtime. Keep the existing loader
# path intact and append the runtime shipped by nvidia-cuda-runtime-cu12.
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

prefill_url="http://${HOST}:${PREFILL_PORT}"
decode_url="http://${HOST}:${DECODE_PORT}"
proxy_url="http://${HOST}:${PROXY_PORT}"
bootstrap_url="http://${HOST}:${BOOTSTRAP_PORT}"
prefill_engine_id="${ENGINE_ID_PREFIX}-prefill"
decode_engine_id="${ENGINE_ID_PREFIX}-decode"

graph_args=()
if [[ "$ENABLE_GRAPH" == 1 || "$ENABLE_GRAPH" == true || "$ENABLE_GRAPH" == yes ]]; then
  graph_args=(--enable-graph)
fi

start_engine() {
  local role="$1"
  local gpu="$2"
  local port="$3"
  local engine_id="$4"
  local kv_role="$5"
  local kv_config
  kv_config="$(
    printf '{"kv_connector":"MooncakeConnector","engine_id":"%s","kv_role":"%s","kv_connector_extra_config":{"mooncake_protocol":"%s","num_workers":%s}}' \
      "$engine_id" "$kv_role" "$MOONCAKE_PROTOCOL" "$MOONCAKE_WORKERS"
  )"

  echo "[$role] gpu=$gpu http=${HOST}:${port} engine_id=$engine_id"
  CUDA_VISIBLE_DEVICES="$gpu" \
  INFINILM_HOST_IP="$HOST" \
  INFINILM_MOONCAKE_BOOTSTRAP_HOST="$HOST" \
  INFINILM_MOONCAKE_BOOTSTRAP_PORT="$BOOTSTRAP_PORT" \
  "$PYTHON" -m infinilm.server.inference_server \
    --model "$MODEL" \
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
}

start_engine prefill "$PREFILL_GPU" "$PREFILL_PORT" "$prefill_engine_id" kv_producer
start_engine decode "$DECODE_GPU" "$DECODE_PORT" "$decode_engine_id" kv_consumer

wait_for_url "prefill server" "$prefill_url/health" "${pids[0]}"
wait_for_url "decode server" "$decode_url/health" "${pids[1]}"
wait_for_url "Mooncake bootstrap" "$bootstrap_url/query" "${pids[0]}"

echo "[proxy] http=${HOST}:${PROXY_PORT}"
"$PYTHON" -m infinilm.server.mooncake_proxy_server \
  --host "$HOST" \
  --port "$PROXY_PORT" \
  --prefill "$prefill_url" "$BOOTSTRAP_PORT" \
  --decode "$decode_url" &
pids+=("$!")
wait_for_url "PD proxy" "$proxy_url/openapi.json" "${pids[2]}"

cat <<EOF
PD API:       $proxy_url/v1
Prefill:      $prefill_url
Decode:       $decode_url
Bootstrap:    $bootstrap_url
Transfer:     mooncake/$MOONCAKE_PROTOCOL
Press Ctrl+C to stop all services.
EOF

wait -n "${pids[@]}"
status=$?
fail "a PD service exited unexpectedly (status=$status)"
