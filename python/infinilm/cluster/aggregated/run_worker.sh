cd /workspace_codex/InfiniLM

RUN_DIR=/tmp/infinilm-single-worker
mkdir -p "$RUN_DIR/discovery"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/python" \
python -m infinilm.cluster.worker \
  --model /data-aisoft/mechdancer/models/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --namespace qwen-single \
  --discovery-backend file \
  --file-kv "$RUN_DIR/discovery" \
  --advertise-host 127.0.0.1 \
  --host 127.0.0.1 \
  --port 18081 \
  --device nvidia \
  --tp 1 \
  --block-size 256 \
  --max-new-tokens 2644 \
  --num-blocks 100 \
  --max-batch-size 32 \
  --log-level INFO \
  --enable-paged-attn \
  --attn flash-attn \
  --enable-graph
