# TCP request plane 设计存档

本文保存早期 TCP request plane 版本的实现思路。当前代码固定使用 HTTP
请求面，也不再提供 `--request-plane` 参数。

## 1. 目标架构

TCP 版本采用控制面与数据面分离：

```text
control plane: file discovery
  worker  --register/heartbeat--> discovery/*.json
  frontend <--periodic snapshot-- discovery/*.json

data plane: raw TCP JSON lines
  frontend -> TcpWorkerClient -> TcpWorkerService -> AsyncLLMEngine
```

完整请求链路如下：

```text
client
  -> frontend /v1/chat/completions
  -> AggregatedFrontend
  -> RoundRobinRouter
  -> TcpWorkerClient
  -> raw TCP JSON-lines
  -> TcpWorkerService
  -> InferenceServer helpers
  -> AsyncLLMEngine.add_chat_request()
  -> AsyncLLMEngine.stream_request()
  -> TcpWorkerService delta/finish/error frames
  -> frontend OpenAI JSON/SSE
```

该设计参考了 Dynamo 的思路：file registry 只承担服务发现和租约状态，
不承载推理 payload；实际推理数据和 token 流走独立请求面。

## 2. 模块职责

### `protocol.py`

定义 TCP 线协议和握手数据：

```python
PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 16 * 1024 * 1024
```

帧格式是一行 UTF-8 JSON：

```text
json.dumps(message) + "\n"
```

`MAX_FRAME_BYTES` 限制单行大小，避免异常客户端发送无限长行。

### `tcp.py`

frontend 侧 TCP 客户端，核心是 `TcpWorkerClient`：

- `connect()`：建连、发送 `hello`、校验 `hello_ack`。
- `health()`：独立连接上发送 health 帧并等待响应。
- `stream_chat()`：发送 chat 帧，持续读取 `accepted/delta/finish/error`。
- `close(request_id)`：可选发送 cancel 帧，然后关闭连接。

客户端是一请求一连接模型，不在多个 frontend 请求间复用 TCP 连接，
避免多路请求共享同一个 reader/writer 时需要额外多路复用协议。

### `worker_service.py`

worker 侧 `TcpWorkerService`：

1. `asyncio.start_server()` 监听 raw TCP 端口。
2. 若测试传入 port 0，从 socket 回填真实端口。
3. 将 `WorkerRecord` 写入 file discovery。
4. 启动 heartbeat 后台任务。
5. 每条连接先做协议版本握手。
6. 按帧处理 `health/chat/cancel`。
7. chat 创建后台生成任务，把 token 输出写回 TCP。
8. 连接关闭或 cancel 时取消生成任务并 abort engine request。

### `router.py`

`RoundRobinRouter` 负责：

- 从 `FileServiceWatcher` 获取可用 worker。
- 按 round-robin 选择 worker。
- 对每个请求创建独立 `TcpWorkerClient`。
- 首个输出前遇到连接/超时错误时切换 worker。
- 已输出 token 后失败则直接抛错，避免换 worker 后重复生成。
- 统计 requests、tokens、errors 和每个 worker 的请求数。

## 3. TCP 握手协议

frontend 第一帧：

```json
{
  "op": "hello",
  "protocol_version": 1,
  "worker_type": "aggregated"
}
```

worker 响应：

```json
{
  "type": "hello_ack",
  "protocol_version": 1,
  "worker_type": "aggregated",
  "served_model_name": "Qwen3-0.6B"
}
```

版本不一致时连接失败。这样可以在后续协议不兼容时升级
`PROTOCOL_VERSION`，避免旧 frontend 与新 worker 产生歧义解析。

## 4. health 会话

health 使用独立 TCP 连接：

```text
connect -> hello -> hello_ack
  -> {"op":"health"}
  <- {"type":"health","status":"ready","engine_healthy":true,...}
  -> close
```

`engine_healthy` 来自 `AsyncLLMEngine.is_healthy()`。

## 5. chat 生成流程

frontend 发送：

```json
{
  "op": "chat",
  "request_id": "cmpl-...",
  "payload": {
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }
}
```

worker 首先返回：

```json
{"type": "accepted", "request_id": "cmpl-..."}
```

然后调用：

```python
messages = inference_server._normalize_messages(payload["messages"])
sampling_params = inference_server._build_sampling_params(payload)
req = engine.add_chat_request(...)
async for token_output in engine.stream_request(req):
    ...
```

每个非 EOS token 返回：

```json
{
  "type": "delta",
  "request_id": "cmpl-...",
  "content": "token text"
}
```

完成时返回：

```json
{
  "type": "finish",
  "request_id": "cmpl-...",
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 2,
    "total_tokens": 5
  }
}
```

异常时返回：

```json
{
  "type": "error",
  "request_id": "cmpl-...",
  "message": "..."
}
```

## 6. 取消流程

frontend 流式生成器被关闭时执行：

```text
stream.aclose()
  -> TcpWorkerClient.close(request_id)
  -> send {"op":"cancel","request_id":"..."}
  -> close TCP socket
```

worker 收到 cancel 后：

```text
requests[request_id].cancel()
  -> TcpWorkerService._generate() CancelledError
  -> engine.add_aborted_req(req, FinishReason.CANCELED)
```

如果 frontend 直接断开 TCP，worker 的连接循环结束，finally 也会取消该连接上
所有未完成生成任务，并调用 `add_aborted_req()`。

## 7. 与 file discovery 的关系

`TcpWorkerService.start()` 的顺序是：

```text
start TCP server
  -> read actual bound port
  -> registry.register(record)
  -> create heartbeat task
```

心跳记录包含：

- `registered_at`：首次注册时间，只用于观测。
- `heartbeat_at`：最近心跳时间。
- `lease_expires_at`：租约截止时间。
- `host/port`：TCP worker 地址。
- `transport: "tcp"`：请求面类型。
- `engine_healthy`：engine 健康状态。

frontend 的 `FileServiceWatcher` 周期性 snapshot：

```text
heartbeat 时间刷新
  -> lease_expires_at = heartbeat_at + lease_seconds

worker 停止心跳
  -> lease_expires_at < now
  -> snapshot 删除过期 JSON
  -> router 不再把该 worker 作为候选
```

## 8. 该实现的取舍

优点：

- 数据面协议简单，单帧 JSON 容易调试。
- 显式 cancel 帧，取消语义直接。
- 不依赖 OpenAI SDK 或 HTTP/SSE 中间层。
- 请求和连接生命周期一一对应，便于隔离故障。

限制：

- 需要自维护握手、帧编码、帧大小、连接生命周期和取消协议。
- frontend 与 worker 之间不是标准 OpenAI-compatible API。
- 连接不跨请求复用，缺少 HTTP 生态中现成的连接池、超时和观测工具。
- 后续扩展多路复用、认证、压缩、代理时都要自定义协议。

这些限制是收敛到 HTTP 请求面的主要原因。HTTP 版本继续复用 file discovery
控制面，但数据面改为：

```text
HttpWorkerClient -> AsyncOpenAI -> httpx.AsyncClient
  -> worker /v1/chat/completions
```
