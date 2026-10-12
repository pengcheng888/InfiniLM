# Dynamo Qwen3-0.6B 三副本聚合式推理复现指南

本文记录在单机 GPU 0-2 上启动 3 个独立 vLLM 副本，并由 Dynamo Frontend 提供统一 OpenAI API 与轮询路由的完整步骤。每个副本同时执行 prefill 和 decode，单卡占用约 30% A100 80GB 显存。

## 目标拓扑

| 项目 | 值 |
| --- | --- |
| 模型路径 | `/data-aisoft/mechdancer/models/Qwen3-0.6B` |
| 对外模型名 | `Qwen3-0.6B` |
| Namespace | `qwen3-agg` |
| 服务发现 | 本机 `file` backend |
| Request plane | `tcp` |
| 路由模式 | `round-robin` |
| Frontend 地址 | `http://127.0.0.1:18000` |
| Worker 系统端口 | `18081`, `18082`, `18083` |
| GPU 映射 | GPU 0/1/2 各一个 worker |
| 显存占比 | `--gpu-memory-utilization 0.3` |

## 1. 预检查

在 Dynamo 仓库根目录执行：

```bash
cd /workspace_codex/dynamo
```

确认 GPU 0-2 可见且显存满足要求：

```bash
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used \
  --format=csv,noheader
```

确认端口未被占用。`18000` 是 Frontend API 端口，`18081-18083` 是 worker 系统端口：

```bash
ss -ltnp | grep -E ':(18000|18081|18082|18083)\b' || true
```

确认模型文件存在：

```bash
test -f /data-aisoft/mechdancer/models/Qwen3-0.6B/config.json
test -f /data-aisoft/mechdancer/models/Qwen3-0.6B/model.safetensors
du -sh /data-aisoft/mechdancer/models/Qwen3-0.6B
```

确认 Dynamo 和 vLLM 已安装：

```bash
python3 -c 'import dynamo, vllm; print("dynamo:", dynamo.__path__); print("vllm:", vllm.__version__)'
```

## 2. 创建运行目录

每个独立启动批次使用自己的 file discovery 目录，避免与旧实例互相发现：

```bash
RUN_DIR="/tmp/dynamo-qwen3-agg-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${RUN_DIR}/discovery"
echo "RUN_DIR=${RUN_DIR}"
```

后续所有 Frontend 和 worker 命令都在同一个 shell 会话中执行，保证共享同一个 `RUN_DIR`。

## 3. 启动 Frontend

Frontend 必须绕过本机地址的 HTTP 代理；此前验证时如果未设置 `NO_PROXY`，Frontend 访问 worker 元数据会返回 502。

```bash
DYN_FILE_KV="${RUN_DIR}/discovery" \
NO_PROXY='localhost,127.0.0.1,::1,192.168.162.61' \
no_proxy='localhost,127.0.0.1,::1,192.168.162.61' \
nohup python3 -m dynamo.frontend \
  --namespace qwen3-agg \
  --discovery-backend file \
  --request-plane tcp \
  --router-mode round-robin \
  --router-min-initial-workers 3 \
  --http-host 127.0.0.1 \
  --http-port 18000 \
  >"${RUN_DIR}/frontend.log" 2>&1 &

echo $! >"${RUN_DIR}/frontend.pid"
```

查看启动日志：

```bash
tail -f "${RUN_DIR}/frontend.log"
```

## 4. 启动 3 个 vLLM worker

分别在 3 个终端执行以下命令也可以；如果在同一个 shell 中连续粘贴，保持 `RUN_DIR` 和 `DYN_FILE_KV` 不变即可。

### GPU 0

```bash
DYN_FILE_KV="${RUN_DIR}/discovery" \
CUDA_VISIBLE_DEVICES=0 \
DYN_SYSTEM_PORT=18081 \
nohup python3 -m dynamo.vllm \
  --model /data-aisoft/mechdancer/models/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --namespace qwen3-agg \
  --discovery-backend file \
  --request-plane tcp \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.3 \
  --enforce-eager \
  --kv-events-config '{"enable_kv_cache_events":false}' \
  >"${RUN_DIR}/worker-gpu0.log" 2>&1 &

echo $! >"${RUN_DIR}/worker-gpu0.pid"
```

### GPU 1

```bash
DYN_FILE_KV="${RUN_DIR}/discovery" \
CUDA_VISIBLE_DEVICES=1 \
DYN_SYSTEM_PORT=18082 \
nohup python3 -m dynamo.vllm \
  --model /data-aisoft/mechdancer/models/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --namespace qwen3-agg \
  --discovery-backend file \
  --request-plane tcp \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.3 \
  --enforce-eager \
  --kv-events-config '{"enable_kv_cache_events":false}' \
  >"${RUN_DIR}/worker-gpu1.log" 2>&1 &

echo $! >"${RUN_DIR}/worker-gpu1.pid"
```

### GPU 2

```bash
DYN_FILE_KV="${RUN_DIR}/discovery" \
CUDA_VISIBLE_DEVICES=2 \
DYN_SYSTEM_PORT=18083 \
nohup python3 -m dynamo.vllm \
  --model /data-aisoft/mechdancer/models/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --namespace qwen3-agg \
  --discovery-backend file \
  --request-plane tcp \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.3 \
  --enforce-eager \
  --kv-events-config '{"enable_kv_cache_events":false}' \
  >"${RUN_DIR}/worker-gpu2.log" 2>&1 &

echo $! >"${RUN_DIR}/worker-gpu2.pid"
```

查看某个 worker 的加载过程：

```bash
tail -f "${RUN_DIR}/worker-gpu0.log"
```

## 5. 等待并验证服务

### 5.1 检查 Frontend

```bash
curl --fail --show-error http://127.0.0.1:18000/health
```

### 5.2 检查 3 个 worker

```bash
for port in 18081 18082 18083; do
  echo "worker ${port}"
  curl --fail --silent "http://127.0.0.1:${port}/health" | jq .
done
```

每个 worker 的 `.status` 都应为 `ready`。

### 5.3 检查模型拓扑

```bash
curl --fail --silent \
  http://127.0.0.1:18000/v1/models/Qwen3-0.6B/ready |
  jq .
```

关键结果：

- 顶层 `ready` 为 `true`
- `qwen3-agg` namespace 为 ready
- 聚合式 worker 数量为 `3`

说明：聚合式 worker 在部分拓扑接口中显示为 `decode`，这是 Dynamo 对 worker type 的命名；本配置中每个 worker 实际同时执行 prefill 和 decode。

### 5.4 确认 GPU 进程

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
  --format=csv,noheader
```

应看到 3 个 vLLM EngineCore 进程，分别对应 GPU 0、1、2。按 `0.3` 显存占比配置，A100 80GB 上每张卡约占用 24-25 GiB。

## 6. 推理冒烟测试

使用仓库自带检查脚本：

```bash
cd /workspace_codex/dynamo

python3 .agents/skills/dynamo-router-starter/scripts/check_router_health.py \
  --base-url http://127.0.0.1:18000 \
  --model Qwen3-0.6B \
  --prompt '只回复 READY' \
  --max-tokens 32 \
  --retries 10
```

成功条件：

- 输出包含 `"ok": true`
- `/v1/models` 返回 HTTP 200
- `/v1/chat/completions` 返回 HTTP 200

也可以直接调用 OpenAI-compatible API：

```bash
curl --silent --show-error --fail \
  http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Return the single word READY."}
    ],
    "temperature": 0,
    "max_tokens": 8,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }' | jq .
```

## 7. 验证 round-robin 覆盖 3 个 worker

连续发送 6 个串行请求：

```bash
for i in 1 2 3 4 5 6; do
  echo "request ${i}"
  curl --fail --silent \
    http://127.0.0.1:18000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "Qwen3-0.6B",
      "messages": [
        {"role": "user", "content": "Return the single word READY."}
      ],
      "temperature": 0,
      "max_tokens": 8,
      "chat_template_kwargs": {
        "enable_thinking": false
      }
    }' | jq -e '.choices | length > 0' > /dev/null
done

echo "6 个请求全部成功"
```

查看 3 个 worker 的请求指标：

```bash
for port in 18081 18082 18083; do
  echo "===== worker ${port} ====="
  curl --silent "http://127.0.0.1:${port}/metrics" |
    grep -E '^vllm:request_success_total'
done
```

如果指标名为空，先查看可用指标：

```bash
curl --silent http://127.0.0.1:18081/metrics | grep -E 'request.*(success|total)' | head
```

此前实际验证中，6 个串行请求后 3 个 worker 的成功请求计数分别为 `2、2、2`，说明 round-robin 覆盖了全部副本。

## 8. 排障

### 查看进程

```bash
pgrep -af 'dynamo\.(frontend|vllm)'
```

### 查看日志

```bash
tail -100 "${RUN_DIR}/frontend.log"
tail -100 "${RUN_DIR}/worker-gpu0.log"
tail -100 "${RUN_DIR}/worker-gpu1.log"
tail -100 "${RUN_DIR}/worker-gpu2.log"
```

### 常见问题

- `8000` 被其他用户进程占用时，本方案使用 `18000`；不要直接终止不属于本次启动的进程。
- Frontend 访问 worker 元数据返回 502 时，检查 `NO_PROXY` 和 `no_proxy` 是否包含 `localhost`、`127.0.0.1`、`::1` 和本机内网 IP。
- worker 无法互相发现时，确认 Frontend 与 3 个 worker 使用完全相同的 `DYN_FILE_KV`。
- 三个 `DYN_SYSTEM_PORT` 必须互不相同。
- `--enforce-eager` 便于功能验证；正式吞吐测试时可移除该参数，让 vLLM 使用 CUDA Graph。
- 本流程是功能冒烟测试，不能作为吞吐量或延迟基准结果。

## 9. 停止本次启动的服务

优先使用启动时写入的 PID 文件，只停止本批次进程：

```bash
for pid_file in \
  "${RUN_DIR}/frontend.pid" \
  "${RUN_DIR}/worker-gpu0.pid" \
  "${RUN_DIR}/worker-gpu1.pid" \
  "${RUN_DIR}/worker-gpu2.pid"; do
  if [[ -s "${pid_file}" ]]; then
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}"
    fi
  fi
done
```

等待进程退出后确认：

```bash
sleep 5
pgrep -af 'dynamo\.(frontend|vllm)' || true
```

不要使用宽泛的 `pkill` 清理服务，以免影响不属于本批次的其他 Frontend 或 worker。
