# InfiniLM Cluster

`infinilm.cluster` 是 InfiniLM 的多副本聚合式推理服务层。它在现有
`InferenceServer` 和 `AsyncLLMEngine` 之上增加两层能力：

1. **file discovery 控制面**：worker 把自己的 HTTP 地址、模型名和健康状态写入
   本机 JSON 文件，frontend 周期性读取这些文件。
2. **HTTP 数据面**：frontend 对外提供统一 OpenAI API，并在多个 worker 之间
   做 round-robin / least-loaded / session-affinity / kv-aware 路由、超时控制、
   失败切换和流式转发。

当前运行时只支持：

```text
--discovery-backend file
--request-plane http
```

每个 worker 都是一个完整的聚合式推理实例：prefill 和 decode 都在本 worker 内完成。
集群层不做跨 worker KV cache 迁移，也不做张量切分。

## 1. 总体架构

```text
                +---------------------------+
 client API --> | ClusterFrontend / FastAPI |
                +---------------------------+
                          |
                          | OpenAI-compatible HTTP/SSE
                          v
                +---------------------------+
                | ClusterRouter          |
                | - worker 选择             |
                | - request timeout         |
                | - 首 token 前故障切换     |
                | - metrics                 |
                +---------------------------+
                          |
                          | HttpWorkerClientFactory
                          |   -> HttpWorkerClient
                          |   -> AsyncOpenAI
                          |   -> shared httpx.AsyncClient
                          v
        +----------------+----------------+----------------+
        |                |                |
        v                v                v
 +-------------+  +-------------+  +-------------+
 | worker 0    |  | worker 1    |  | worker 2    |
 | Uvicorn API |  | Uvicorn API |  | Uvicorn API |
 | LLM engine  |  | LLM engine  |  | LLM engine  |
 +-------------+  +-------------+  +-------------+
        |                |                |
        +----------------+----------------+
                         |
                         v
              +--------------------------+
              | file discovery directory |
              | namespace...instance.json |
              +--------------------------+
```

控制面和数据面是分离的：

```text
control plane:
  worker -> discovery JSON -> frontend watcher

data plane:
  client -> frontend -> router -> worker HTTP API -> AsyncLLMEngine
```

discovery 文件只保存服务元数据，不承载推理请求和 token 流。

## 2. 目录与模块

### `config.py`

负责 frontend 和 worker 的 cluster 参数解析。

主要公共配置：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--namespace` | `infinilm-agg` | discovery 命名空间，frontend 和 worker 必须一致 |
| `--discovery-backend` | `file` | 当前仅支持文件注册表 |
| `--file-kv` | `/tmp/infinilm_store_kv` | discovery JSON 根目录 |
| `--request-plane` | `http` | 当前仅支持 HTTP |
| `--registry-ttl` | `10` 秒 | worker 心跳租约时长 |
| `--heartbeat-interval` | `2` 秒 | worker 心跳间隔 |

frontend 专属参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--router-mode` | `round-robin` | 支持 `round-robin`、`least-loaded`、`session-affinity`、`kv-aware` |
| `--kv-router-min-tokens` | `1` | `kv-aware` 模式下允许复用候选的最小预测命中 token 数 |
| `--router-affinity-ttl-secs` | `300` 秒 | session-affinity 条目生存时间 |
| `--router-affinity-capacity` | `4096` | session-affinity LRU 表容量 |
| `--router-min-initial-workers` | `1` | frontend 完成启动前需要的最小 worker 数 |
| `--http-host` | `127.0.0.1` | frontend 监听地址 |
| `--http-port` | `18000` | frontend 监听端口 |
| `--connect-timeout` | `3` 秒 | frontend 到 worker 的建连超时 |
| `--request-timeout` | `1000` 秒 | 单次推理请求整体超时 |
| `--health-interval` | `1` 秒 | discovery watcher 轮询间隔 |

worker 专属参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--served-model-name` | 模型目录名 | OpenAI API 中展示的模型名 |
| `--advertise-host` | `127.0.0.1` | 写入 discovery 的 worker 地址 |
| `--host` | `BaseConfig` 默认值 | worker HTTP 服务监听地址 |
| `--port` | `BaseConfig` 默认值 | worker HTTP 服务监听端口 |

`remove_cluster_worker_args()` 会剥离 cluster 专属参数，再把剩余参数交给
`BaseConfig` 解析原有推理配置。

### `registry.py`

实现文件版服务发现。

核心对象：

- `WorkerRecord`：一条 worker 注册记录。
- `FileServiceRegistry`：创建、写入、心跳、注销和 snapshot discovery 记录。
- `FileServiceWatcher`：frontend 侧周期 snapshot，维护当前可用 worker 列表。

注册文件名格式：

```text
{namespace}.{component}.{endpoint}.{instance_id}.json
```

例如：

```text
qwen3.infinilm.generate.12345-abcdef123456.json
```

### `kv_prefix.py`

frontend 侧加载与 worker 相同的 processor，对文本 chat 请求执行 chat
template 和 tokenize，再按 worker 的 block size 计算 chained prefix hash。
多模态请求、无法 tokenize 的请求或 processor 异常会返回 `None`，路由回退到
普通负载策略。

### `cache_index.py`

worker 侧把本地 `BlockManager` 的已发布 prefix hash 导出为本机 JSON 文件：

```text
<file-kv>/cache-index/<namespace>.<instance_id>/
  prefix-cache.snapshot.json
  prefix-cache.events.jsonl
```

snapshot 是权威全量状态，events 是增量 append-only 日志。事件类型包括
`published`、`touched` 和 `evicted`。事件达到默认 `10000` 条后导出器会重写
snapshot 并清空 event 文件；frontend 检测到 event 文件缩短后会重新加载
snapshot。

### `kv_index.py`

frontend 侧维护每个 ready worker 的本地 prefix index。它读取 worker
snapshot，tail event 文件，并在 worker 退出、DRAINING 或元数据不可用时删除
对应索引。路由时只统计连续 prefix hash 匹配的块数。

### `worker_discovery.py`

`WorkerDiscoveryService` 只负责 worker 侧 discovery 生命周期：

```text
start()
  -> registry.register(record)
  -> create heartbeat task

heartbeat loop
  -> registry.heartbeat(...)
  -> 刷新 heartbeat_at / lease_expires_at / engine_healthy

stop()
  -> cancel heartbeat task
  -> registry.unregister(record)
```

它不处理推理请求。真正提供 `/v1/chat/completions` 的是 `ClusterInferenceServer`
继承的 `InferenceServer`。

### `worker.py`

定义 `ClusterInferenceServer` 和 `python -m infinilm.cluster.worker` 入口。

它复用现有 `InferenceServer` 的 HTTP API 和模型初始化逻辑，并在 FastAPI
lifespan 中加入 discovery：

```text
Uvicorn startup
  -> InferenceServer lifespan
  -> AsyncLLMEngine 初始化和启动
  -> ClusterInferenceServer._create_discovery_service(engine)
  -> WorkerDiscoveryService.start()
  -> application ready

Uvicorn shutdown
  -> WorkerDiscoveryService.stop()
  -> AsyncLLMEngine stop
```

这样可以保证 worker 只有在 engine ready 后才被 frontend 发现。

### `http_worker_client_factory.py`

包含两层对象。

`HttpWorkerClientFactory`：

- 持有一个 frontend 共享的 `httpx.AsyncClient` 连接池；
- 配置 connect/read/write/pool timeout；
- 根据 `WorkerRecord` 创建 `HttpWorkerClient`；
- 在 frontend 退出时释放自己拥有的连接池。

`HttpWorkerClient`：

- 绑定一个 worker 的 `http://{host}:{port}/v1`；
- 内部创建 `AsyncOpenAI`；
- 把 OpenAI request 转发给 worker；
- 把 OpenAI 非流式 response 或流式 SSE 转换成 router 内部帧。

OpenAI SDK 设置：

```python
max_retries=0
```

原因是 worker 级重试由 `ClusterRouter` 统一处理，避免 SDK 和 router 双层重试。

### `router.py`

`ClusterRouter` 是 frontend 的数据面路由核心。

职责：

1. 从 `FileServiceWatcher` 获取当前可用 worker；
2. 过滤 `transport=http`、模型名匹配、engine 健康且租约未过期的 worker；
3. 按 `round-robin`、`least-loaded`、`session-affinity` 或 `kv-aware`
   策略选择 worker；
4. 为本次请求创建 `HttpWorkerClient`；
5. 对内部帧流施加整体 deadline；
6. 记录请求数、token chunk 数、错误数和 inflight；
7. 在安全边界内切换 worker。

重试边界：

```text
未产生任何输出 token
  -> 连接错误 / 超时 / HTTP 5xx
  -> 尝试下一个 worker

已经产生输出 token
  -> 不再切换 worker
  -> 直接向客户端暴露错误
```
这样做是为了避免客户端已经收到部分内容后，另一个 worker 重复生成同一请求。

### `frontend.py`

`ClusterFrontend` 对外提供统一 OpenAI API 和运维接口。

API：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | frontend 自身健康状态和当前 worker 数 |
| GET | `/v1/models` | OpenAI models 格式的模型列表 |
| GET | `/v1/models/{model}/ready` | 指定模型是否满足最小副本数 |
| GET | `/metrics` | Prometheus 文本格式指标 |
| GET | `/monitoring` | 内置 worker 监控网页 |
| GET | `/api/monitoring` | 监控网页使用的 JSON 状态快照 |
| POST | `/v1/chat/completions` | OpenAI-compatible chat API |
| POST | `/chat/completions` | 兼容旧路径 |

`main()` 的启动流程：

```text
parse_frontend_args()
  -> logging.basicConfig()
  -> ClusterFrontend(options)
     -> FileServiceRegistry
     -> FileServiceWatcher
     -> ClusterRouter
     -> HttpWorkerClientFactory
     -> FrontendMetrics
  -> uvicorn.run(frontend.create_app())
```

FastAPI lifespan：

```text
refresh file discovery
  -> while worker_count < router_min_initial_workers:
       sleep and refresh
  -> FileServiceWatcher.start(periodic refresh)
  -> serve app
  -> shutdown:
       FileServiceWatcher.stop()
       ClusterRouter.aclose()
```

### `metrics.py`

维护内存版路由指标，并通过 `/metrics` 输出。

指标：

```text
infinilm_router_workers
infinilm_router_requests_total
infinilm_router_errors_total
infinilm_router_tokens_total
infinilm_router_worker_requests_total{worker_id=...}
infinilm_router_worker_errors_total{worker_id=...}
infinilm_router_kv_aware_requests_total
infinilm_router_kv_hit_requests_total
infinilm_router_kv_miss_requests_total
infinilm_router_kv_predicted_saved_tokens_total
infinilm_router_kv_actual_saved_tokens_total
infinilm_router_kv_index_entries
```

`tokens_total` 统计的是 router 输出的内部 token chunk 数，不是 tokenizer 的
严格 usage token 数。非流式响应中的 usage 来自 worker 返回的 OpenAI usage。

### `monitoring/`

监控能力集中在本机 `infinilm.cluster.monitoring` 包中：

```text
monitoring/
  __init__.py
  snapshot.py
  page.py
```

`ClusterFrontend.create_app()` 会同时注册：

```text
GET /monitoring
GET /api/monitoring
```

监控页面没有独立进程和独立端口，随 frontend 的 FastAPI/Uvicorn 生命周期一起
启动和退出。页面使用原生 HTML/CSS/JS，不依赖 CDN、npm 包或前端构建工具，
每 2 秒请求一次 `/api/monitoring`。

页面展示：

- namespace、router mode、ready 状态和 worker 数；
- requests、successful requests、streaming/non-streaming requests、errors、
  retries、inflight、usage tokens 和 token chunks；
- affinity requests、hits、misses、TTL/LRU 状态和活跃 affinity 条目；
- kv-aware 请求数、预测命中、实际命中和全局 prefix index 条目；
- worker 侧 received、active、success、failed、canceled 计数；
- 直接访问 worker 造成的 direct / untracked 请求估计；
- 全局 request duration、streaming TTFT 和 inter-token latency 的
  avg/p50/p95/p99；
- 每个 worker 的 endpoint、GPU、TP、ready/draining 状态、heartbeat 年龄、
  load score、frontend inflight、worker active、lease 剩余时间、
  waiting/running、KV usage、本地 cache hits/hit tokens/prefill tokens、
  frontend 路由数、worker 收到数、direct 数、usage tokens、错误数和最近成功时间。

页面提供“自动刷新”和“刷新”控制。自动刷新默认开启，页面隐藏时暂停，
恢复可见后继续；接口失败时逐步退避，且旧响应不会覆盖新响应。

指标语义：

- `infinilm_router_*` 请求、错误、重试和 usage tokens 只统计经过 frontend
  router 的请求。
- `infinilm_worker_*` 来自 worker HTTP 进程，包含 frontend 路由请求和直接访问
  worker 的请求。
- direct / untracked 是按 worker 估算：
  `max(0, worker_received_requests_total - router_worker_requests_total)`。
- `token chunks`、TTFT 和 inter-token latency 只统计流式请求。
- 非流式请求会更新请求数、request duration 和 worker 返回的 usage tokens。

访问地址：

```text
http://127.0.0.1:18000/monitoring
```

如果 frontend 监听 `0.0.0.0`，监控页面也会暴露到对应网络。生产环境需要在
frontend 前增加访问控制，或仅监听本机地址。

### `TCP_REQUEST_PLANE_DESIGN.md`

这是历史设计存档，记录早期 raw TCP JSON-lines 请求面的协议、握手、取消、
心跳和模块设计。当前运行时代码已经移除 TCP 请求面，该文档不描述当前
生产路径。

### `run_cluster.sh`

这是一个本机启动示例，当前默认启动 1 个 worker：

```text
frontend: 18000
worker:   18081
GPU:      0
model:    Qwen3-0.6B
router:   round-robin
```

默认路由模式为 `round-robin`。`ROUTER_MODE` 环境变量可以覆盖为
`least-loaded`、`session-affinity` 或 `kv-aware`，例如：

```bash
ROUTER_MODE=least-loaded bash python/infinilm/cluster/run_cluster.sh
ROUTER_MODE=kv-aware bash python/infinilm/cluster/run_cluster.sh
```

脚本会同时打印 OpenAI API 地址和监控网页地址。frontend 自身启动时也会
在日志中输出：

```text
Monitoring page: http://127.0.0.1:18000/monitoring
```

如需三副本，可将脚本中的 `GPUS` 默认值恢复为 `0 1 2`，或通过环境变量
`GPUS="0 1 2"` 覆盖。

`*.pyc` 和 `__pycache__` 是 Python 生成产物，不属于源码模块。

## 3. Discovery 数据模型

`WorkerRecord` 的关键字段：

| 字段 | 说明 |
|---|---|
| `namespace` | discovery 命名空间 |
| `component` | 固定为 `infinilm` |
| `endpoint` | 固定为 `generate` |
| `instance_id` | worker 实例唯一 ID |
| `worker_type` | 当前为 `aggregated` |
| `model` | 本地模型路径 |
| `served_model_name` | OpenAI API 中的模型名 |
| `host` | worker 对外 advertised 地址 |
| `port` | worker HTTP 端口 |
| `registered_at` | 首次注册时间，仅用于观测 |
| `heartbeat_at` | 最近一次心跳时间 |
| `lease_expires_at` | 当前租约截止时间 |
| `engine_healthy` | `AsyncLLMEngine.is_healthy()` 的结果 |
| `transport` | 当前固定为 `http` |
| `state` | `ready` 或 `draining`；router 只选择 `ready` |
| `metadata` | GPU、TP、端口、batch size、`load` 负载快照等补充信息 |

时间语义：

```text
lease_expires_at = heartbeat_at + registry_ttl
```

worker 正常心跳时会持续刷新：

```text
heartbeat_at
lease_expires_at
engine_healthy
state
metadata.load
metadata.traffic
```

`registered_at` 不会随 heartbeat 更新。

`metadata.traffic` 来自 worker HTTP 进程，包含直接访问 worker 的流量：

```json
{
  "received_requests_total": 100,
  "streaming_requests_total": 50,
  "non_streaming_requests_total": 50,
  "successful_requests_total": 96,
  "failed_requests_total": 2,
  "canceled_requests_total": 2,
  "active_requests": 3,
  "local_cache_hit_requests_total": 20,
  "local_cache_miss_requests_total": 80,
  "local_cache_hit_tokens_total": 5120,
  "local_prefill_tokens_total": 20480
}
```

## 4. File Registry 流程

### 4.1 注册

```text
WorkerRecord
  -> registry.new_record(...)
  -> registry.register(record)
  -> _write(record)
  -> discovery/*.json
```

写入方式：

1. 先写同目录临时 `.tmp` 文件；
2. 调用 `os.replace()` 原子替换正式 JSON；
3. frontend 只读取 `*.json`。

因此 frontend 不会读到半截 JSON。

### 4.2 心跳

```text
WorkerDiscoveryService._heartbeat()
  -> sleep(heartbeat_interval)
  -> registry.heartbeat(record, engine_healthy=engine.is_healthy())
  -> 原子重写 discovery JSON
```

如果 `engine_healthy=false`，记录仍可能存在，但 router 会把它从可用候选中过滤掉。

### 4.3 DRAINING 与优雅下线

worker 收到 `SIGTERM` 或 `SIGINT` 后，`DrainingUvicornServer` 会先执行：

```text
WorkerDiscoveryService.start_draining()
  -> WorkerRecord.state = draining
  -> 原子写入 discovery
  -> frontend watcher 观察到 draining
  -> ClusterRouter 不再选择该 worker
```

随后 worker 等待 `--drain-grace-seconds`，默认 1 秒。该宽限时间用于覆盖
frontend watcher 轮询间隔，让在途路由决策有时间收敛；宽限期结束后 Uvicorn
进入正常 shutdown，完成存量请求后 unregister。第二次退出信号会跳过宽限，
用于运维强制中断。

### 4.4 Snapshot 与过期删除

frontend watcher 周期性执行：

```text
FileServiceWatcher.run()
  -> registry.snapshot(namespace)
  -> FileServiceRegistry._snapshot()
```

snapshot 时会：

1. 读取所有 `*.json`；
2. 忽略 malformed JSON；
3. 过滤 namespace / endpoint / worker type / engine health；
4. 如果 `lease_expires_at < now`，删除过期 JSON；
5. 按 `instance_id` 排序返回 worker 列表。

如果 worker 被 `kill -9`，没有机会 unregister，注册文件会暂时残留。
frontend 后续 snapshot 到租约过期时会删除它。

## 5. 请求流程

### 5.1 非流式请求

```text
client
  POST /v1/chat/completions
    -> ClusterFrontend.chat_completions()
    -> validate JSON and messages
    -> resolve model
    -> generate request_id
    -> ClusterRouter.stream_chat()
    -> HttpWorkerClient.stream_chat()
    -> AsyncOpenAI.chat.completions.create(stream=False)
    -> worker /v1/chat/completions
    -> InferenceServer._chat()
    -> AsyncLLMEngine
    -> OpenAI chat.completion response
    -> internal finish frame
    -> frontend OpenAI JSON response
```

HTTP worker 非流式响应会被转换成内部帧：

```json
{
  "type": "finish",
  "content": "Ready.",
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 2,
    "total_tokens": 5
  }
}
```

### 5.2 流式请求

```text
client
  POST /v1/chat/completions with stream=true
    -> ClusterFrontend
    -> StreamingResponse(_stream_response)
    -> ClusterRouter.stream_chat()
    -> HttpWorkerClient.stream_chat()
    -> AsyncOpenAI.chat.completions.create(stream=True)
    -> worker SSE
    -> InferenceServer._stream_chat()
    -> AsyncLLMEngine.stream_request()
    -> internal delta/finish frames
    -> frontend OpenAI chat.completion.chunk SSE
    -> [DONE]
```

内部帧：

```json
{"type": "delta", "content": "Ready"}
{"type": "delta", "content": "."}
{"type": "finish", "finish_reason": "stop", "usage": {}}
```

frontend 输出：

```text
data: {"object":"chat.completion.chunk", ...}

data: [DONE]
```

### 5.3 客户端取消

流式取消链路：

```text
client disconnect
  -> frontend StreamingResponse cancelled
  -> router internal stream closed
  -> HttpWorkerClient closes AsyncOpenAI stream
  -> worker HTTP connection disconnect
  -> InferenceServer detects request disconnect
  -> engine.add_aborted_req(...)
```

因此 HTTP/SSE 断开就是数据面的取消信号。

## 6. Worker 选择与故障切换

router 每次请求都会从 watcher 获取可用 worker：

```text
transport == http
served_model_name == requested model
engine_healthy == true
lease not expired
worker_type == aggregated
```

### 6.1 least-loaded

`least-loaded` 是可显式选择的负载感知路由模式。frontend 为每个 worker
维护当前在本
frontend 上的在途请求数；worker heartbeat 会调用
`AsyncLLMEngine.get_load_stats()`，把近似负载快照写入 discovery：

```json
{
  "waiting_requests": 2,
  "running_requests": 3,
  "max_batch_size": 32,
  "kv_capacity": 500,
  "kv_used": 120,
  "kv_usage_ratio": 0.24
}
```

frontend 综合评分：

```text
score =
  0.90 * frontend inflight
+ 0.05 * waiting_requests
+ 0.03 * running_requests / max_batch_size
+ 0.02 * kv_usage_ratio
```

同样选择 score 最小者，平局时轮询。队列和 cache 数值来自 scheduler 的周期性
近似快照，用于路由倾向控制，不作为精确计费或强一致性状态。

### 6.2 round-robin

`round-robin` 不使用负载评分，按当前可用 worker 的稳定顺序轮询：

```python
index = cursor % len(workers)
cursor += 1
```

例如三个 worker 连续六次选择：

```text
A B C A B C
```

该模式适合作为对照基线，或在负载数据不可信、希望请求计数严格均匀时使用。

### 6.3 session-affinity

`session-affinity` 用于尽量把相同会话或相同对话前缀路由到同一个 worker，以
提高该 worker 本地 prefix cache 的复用机会。

affinity key 的来源优先级：

1. `X-InfiniLM-Session-ID` request header；
2. `X-Session-ID` request header；
3. OpenAI request 的 `user` 字段；
4. `model + 首个 system message + 首个 user message` 的稳定前缀指纹。

key 只保存 SHA-256 指纹，不在 frontend 内存或日志中保存原始 session ID。

路由规则：

```text
affinity worker 存在且 ready
  -> 若其 load score 不高于当前最小值 + 1.0，则复用该 worker

affinity worker 不存在、已退出、DRAINING、或明显过载
  -> 回退 least-loaded
  -> 更新 affinity 表
```

affinity 表是每个 frontend 进程内的 LRU：

```text
TTL: --router-affinity-ttl-secs，默认 300 秒
容量: --router-affinity-capacity，默认 4096 条
```

worker 从 discovery 中消失或进入 DRAINING 后，对应 affinity 条目会被清理。
该模式不迁移 KV cache；它只通过稳定路由提高本地 prefix cache 命中概率。

### 6.4 kv-aware

`kv-aware` 面向文本 chat 请求的本地 prefix cache 复用。它不迁移 KV cache，
而是让 frontend 知道每个 worker 已经发布过哪些连续 prefix hash，然后把相同
前缀的新请求发给最可能命中且没有明显过载的 worker。

请求流程：

```text
frontend resolve model
  -> 读取 worker metadata 中的 model path / block size
  -> PrefixHashCalculator tokenize prompt
  -> 计算 namespace + chained block hashes
  -> ClusterRouter.refresh_kv_index()
  -> 每个 worker 连续匹配 hash block
  -> 选择 matched_tokens 最大且负载可接受的 worker
  -> 移除内部 __infinilm_kv_prefix__ 字段后转发
```

worker 必须启用 paged attention、prefix caching，并提供 cache index 文件。
frontend 和 worker 使用相同的模型目录、processor 配置和 block size；namespace
由模型/tokenizer 关键配置文件和 block size 计算得到。

选择规则：

```text
预测 matched_tokens >= --kv-router-min-tokens
且 worker load score <= 当前最小 load score + 1.0
  -> 选择 matched_tokens 最大者，平局按 worker_id 稳定排序

无有效匹配、hash 计算失败、多模态请求、index 不可用或候选 worker 过载
  -> 回退 least-loaded
```

因此 `kv-aware` 是负载感知之上的 cache 倾斜策略，不是绝对亲和。KV cache
可能被 scheduler 淘汰，event tail 和 discovery 心跳也存在短暂延迟；路由决策
是预测性的，最终以 worker 返回的实际 cache 统计为准。

worker 响应中的实际观测值：

```json
{
  "cache": {
    "matched_tokens": 512,
    "prefill_tokens": 18,
    "prompt_tokens": 530
  }
}
```

`matched_tokens` 用于 `infinilm_router_kv_actual_saved_tokens_total` 和
worker cache counters；frontend 预测值对应
`infinilm_router_kv_predicted_saved_tokens_total`。两者接近说明前端索引与
worker 本地 cache 状态一致。

故障切换只发生在首个输出帧之前。对于 HTTP：

- connect error 可重试；
- timeout 可重试；
- worker 返回 5xx 可重试；
worker 返回 4xx 通常表示请求本身不合法，不切换 worker。

如果所有候选 worker 都失败，router 抛出：

```text
all workers failed for request ...
```

frontend 将其转换为服务端错误响应。

## 7. 直接访问单个 worker

frontend 不是 worker 推理的必要依赖。`ClusterInferenceServer` 继承了
`InferenceServer`，因此单独启动 worker 后可以直接访问：

```text
http://127.0.0.1:18081/v1
```

可用接口：

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /chat/completions
```

单 worker 适合调试模型加载、chat template 和引擎输出。多副本统一入口、
round-robin / least-loaded / session-affinity / kv-aware 路由、故障切换和集中指标
仍然由 frontend 提供。

## 8. 启动示例

### 8.1 启动 frontend

```bash
cd /workspace_codex/InfiniLM

PYTHONPATH="$PWD/python" \
python -m infinilm.cluster.frontend \
  --namespace qwen3-agg \
  --discovery-backend file \
  --file-kv /tmp/infinilm-agg/discovery \
  --request-plane http \
  --router-mode session-affinity \
  --router-affinity-ttl-secs 300 \
  --router-affinity-capacity 4096 \
  --router-min-initial-workers 3 \
  --http-host 127.0.0.1 \
  --http-port 18000
```

### 8.2 启动三个 worker

下面是第 0 个 worker 的示例；第 1、2 个 worker 分别使用 GPU `1/2` 和端口
`18082/18083`。

```bash
cd /workspace_codex/InfiniLM

mkdir -p /tmp/infinilm-agg/discovery

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/python" \
python -m infinilm.cluster.worker \
  --model /data-aisoft/mechdancer/models/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --namespace qwen3-agg \
  --discovery-backend file \
  --file-kv /tmp/infinilm-agg/discovery \
  --request-plane http \
  --advertise-host 127.0.0.1 \
  --host 127.0.0.1 \
  --port 18081 \
  --device nvidia \
  --tp 1 \
  --block-size 256 \
  --max-new-tokens 2644 \
  --num-blocks 500 \
  --max-batch-size 32 \
  --log-level INFO \
  --enable-paged-attn \
  --attn flash-attn \
  --enable-graph
```

frontend 和 worker 必须满足：

```text
namespace 相同
file-kv 相同
request-plane 均为 http
```

### 8.3 请求 frontend

非流式：

```bash
curl http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "请最后单独输出一行：READY"}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

流式：

```bash
curl -N http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "请最后单独输出一行：READY"}
    ],
    "max_tokens": 256,
    "temperature": 0,
    "stream": true
  }'
```

readiness：

```bash
curl http://127.0.0.1:18000/v1/models/Qwen3-0.6B/ready
```

指标：

```bash
curl http://127.0.0.1:18000/metrics
```

监控网页：

```text
http://127.0.0.1:18000/monitoring
```

主要指标包括：

```text
infinilm_router_mode
infinilm_router_metrics_source
infinilm_router_requests_total
infinilm_router_attempts_total
infinilm_router_successful_requests_total
infinilm_router_streaming_requests_total
infinilm_router_non_streaming_requests_total
infinilm_router_errors_total
infinilm_router_retries_total
infinilm_router_tokens_total
infinilm_router_usage_prompt_tokens_total
infinilm_router_usage_completion_tokens_total
infinilm_router_usage_total_tokens_total
infinilm_router_inflight
infinilm_router_affinity_requests_total
infinilm_router_affinity_hits_total
infinilm_router_affinity_misses_total
infinilm_router_affinity_expired_total
infinilm_router_affinity_evictions_total
infinilm_router_affinity_unkeyed_requests_total
infinilm_router_affinity_entries
infinilm_router_kv_aware_requests_total
infinilm_router_kv_hit_requests_total
infinilm_router_kv_miss_requests_total
infinilm_router_kv_predicted_saved_tokens_total
infinilm_router_kv_actual_saved_tokens_total
infinilm_router_kv_index_entries
infinilm_router_request_duration_seconds
infinilm_router_ttft_seconds
infinilm_router_inter_token_latency_seconds
infinilm_router_worker_requests_total
infinilm_router_worker_usage_prompt_tokens_total
infinilm_router_worker_usage_completion_tokens_total
infinilm_router_worker_usage_total_tokens_total
infinilm_router_worker_errors_total
infinilm_router_worker_retries_total
infinilm_router_worker_inflight
infinilm_router_worker_load_score
infinilm_router_worker_waiting_requests
infinilm_router_worker_running_requests
infinilm_router_worker_kv_usage_ratio
infinilm_router_worker_last_success_timestamp
infinilm_router_worker_ttft_seconds
infinilm_router_worker_request_duration_seconds
infinilm_worker_state
infinilm_worker_received_requests_total
infinilm_worker_streaming_requests_total
infinilm_worker_non_streaming_requests_total
infinilm_worker_successful_requests_total
infinilm_worker_failed_requests_total
infinilm_worker_canceled_requests_total
infinilm_worker_active_requests
infinilm_worker_untracked_requests_total
infinilm_worker_cache_hit_requests_total
infinilm_worker_cache_miss_requests_total
infinilm_worker_cache_hit_tokens_total
infinilm_worker_prefill_tokens_total
```

## 9. 测试

运行 cluster 单元测试：

```bash
cd /workspace_codex/InfiniLM
PYTHONPATH=python python -m pytest -q test/service/test_cluster.py
```

测试覆盖：

- file registry 注册、snapshot、租约过期和文件删除；
- 默认接受 `round-robin`，显式接受 `least-loaded`，拒绝 `least-inflight`；
- round-robin 三 worker 分配；
- least-loaded 在并发在途请求下避开忙 worker；
- least-loaded 使用 waiting / running / KV pressure 元数据选择轻载 worker；
- session-affinity 的 key 派生、TTL 过期、LRU 淘汰、worker 退出回退和过载回退；
- frontend 与 worker 的 prefix hash 一致性、cache snapshot/event 导出和
  `kv-aware` 最长连续前缀选择；
- frontend 从 `X-Session-ID` 提取 affinity key，且不会把内部 routing key 转发给 worker；
- `AsyncLLMEngine.get_load_stats()` 读取 scheduler 队列与 cache 状态；
- `WorkerDiscoveryService` 注册、心跳和退出清理；
- worker 侧请求计数、直接访问 worker 流量估算；
- worker `DRAINING` 状态和 Uvicorn 优雅下线；
- frontend 到 HTTP worker 的非流式请求；
- OpenAI SSE 到内部 `delta/finish` 帧的转换；
- worker 5xx 后切换下一个 worker。
- TTFT、token 间隔、请求耗时和 retry 指标。
- KV-aware 预测命中、worker 实际命中和 Prometheus 文本格式解析。
- `/monitoring` 页面和 `/api/monitoring` JSON 快照。

单元测试使用 `httpx.MockTransport` 模拟 worker 网络层，不加载模型、不占用 GPU。
真实三副本部署需要另行启动 frontend 和 worker 进行集成验证。

## 10. 设计边界

当前设计面向单机多 GPU / 单机多进程部署：

- file discovery 依赖同一个本机文件系统；
- frontend 和 worker 需要能通过 `advertise-host:port` 互相访问；
- worker API 没有 frontend 级鉴权，生产暴露端口时需要外部访问控制；
- 集群层不聚合多个 worker 的 batch，每个 worker 内部独立调度；
- 集群层不迁移 KV cache；
- pipeline parallel 当前要求 `pp=1`；
- 如果启动多个模型，模型名由 `served_model_name` 区分；
- frontend readiness 只反映 discovery 和租约状态，不主动探测每个 worker 的
  `/health` endpoint。

## 11. 关键类关系

```text
FrontendOptions
  -> ClusterFrontend
       -> FileServiceRegistry
       -> FileServiceWatcher
       -> ClusterRouter
            -> HttpWorkerClientFactory
                 -> HttpWorkerClient
                      -> AsyncOpenAI
                      -> httpx.AsyncClient
            -> PrefixHashCalculator
            -> KvIndexStore
       -> FrontendMetrics

WorkerOptions
  -> ClusterInferenceServer(InferenceServer)
       -> AsyncLLMEngine
       -> FileServiceRegistry
       -> WorkerDiscoveryService
       -> WorkerCacheIndexExporter
```

运行时对象关系：

```text
FileServiceRegistry
  -> WorkerRecord list
  -> FileServiceWatcher.workers
  -> ClusterRouter.available_workers()
  -> HttpWorkerClientFactory.create(worker)
  -> HttpWorkerClient
```
