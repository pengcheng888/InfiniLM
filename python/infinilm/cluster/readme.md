# InfiniLM Cluster

InfiniLM Cluster 支持两种服务形态：

1. **聚合式服务**：每个 worker 都能独立完成 prefill 和 decode，frontend 在多个完整推理实例之间路由请求。
2. **PD 分离服务**：prefill worker 计算 prompt 并发布 KV，decode worker 通过 Mooncake 拉取 KV 并完成增量生成。

两种形态共享 worker registry 控制面。控制面支持本机 file backend 和分布式 etcd backend；frontend 到 worker 的请求面当前使用 HTTP。

## 1. 总体架构

### 聚合式服务

```text
client
  -> aggregated frontend
  -> router
  -> HTTP worker client
  -> aggregated worker
  -> inference server
  -> LLM engine
```

worker 将 HTTP 地址、模型名、健康状态和负载快照注册到 worker registry。frontend 周期读取 registry，并把请求路由给 ready 且 transport 匹配的 worker。

### PD 分离服务

```text
client
  -> PD frontend
  -> prefill worker
       执行 prompt prefill 并发布 KV
  -> Mooncake KV transfer
  -> decode worker
       拉取 KV 并增量生成
  -> client
```

PD frontend 分别选择 prefill 和 decode worker。prefill 请求只生成一个 token，并发布 KV；decode 请求携带 prefill bootstrap 地址和 engine ID，由 decode worker 查询 KV 传输信息。

### 控制面与数据面

```text
control plane:
  worker -> worker registry -> frontend watcher

request plane:
  client -> frontend -> HTTP worker

PD KV plane:
  prefill worker -> Mooncake -> decode worker
```

worker registry 只保存元数据，不承载推理请求和 token 流。

### PD HTTP transport

`disaggregated/pd_http_transport.py` 是 discovery-aware PD frontend 专用的
HTTP 转发层，负责组装 prefill/decode 的 `kv_transfer_params`、添加内部请求
header、透传响应 bytes，并在请求取消时释放 httpx response。该模块不依赖
standalone `mooncake_proxy_server.py`，旧代理服务可以独立演进。

### PD client 生命周期

PD frontend 在请求路径外维护 worker HTTP client：

```text
worker registry
  -> WorkerRegistryWatcher 周期 snapshot
  -> PdClientReconciler 过滤 ready 且 engine healthy 的 P/D record
  -> WorkerClientPoolManager 调度每个 worker 的 bootstrap task
  -> prefill /query 获取 engine ID
  -> 初始化成功后进入 routable pool
  -> /health、/v1/models 和 chat 请求只读取 routable pool
```

`WorkerClientPoolManager` 的状态锁只保护内存中的 desired/pending/pool
状态；`httpx.AsyncClient` 创建、prefill bootstrap 查询和连接关闭都在锁外
执行。同一个 worker 的并发初始化通过 pending task 去重，不同 worker 可以
并发初始化。worker 从 discovery 消失时，pending task 会被取消；已在服务
存量请求的 client 会先标记为不可路由，待 inflight 归零后关闭。

## 2. 配置

### 公共配置

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--namespace` | `infinilm-agg` | frontend 和 worker 共享的 discovery namespace |
| `--discovery-backend` | `file` | `file` 或 `etcd` |
| `--file-kv` | `/tmp/infinilm_store_kv` | file backend 的注册目录 |
| `--etcd-endpoints` | `127.0.0.1:2379` | 逗号分隔的 etcd endpoint |
| `--etcd-prefix` | `/infinilm` | etcd discovery key prefix |
| `--etcd-username` | 空 | 可选 etcd 用户名 |
| `--etcd-password-env` | `ETCD_PASSWORD` | 存放 etcd 密码的环境变量名 |
| `--etcd-ca-cert` | 空 | etcd CA 证书；HTTPS endpoint 必填 |
| `--etcd-cert-file` / `--etcd-key-file` | 空 | 可选 etcd 双向 TLS 客户端证书 |
| `--etcd-timeout` | `3` 秒 | etcd 操作超时 |
| `INFINILM_CLUSTER_CONNECT_TIMEOUT` | `3` 秒 | frontend 到 worker 的建连、写请求和连接池等待超时 |
| `INFINILM_CLUSTER_REQUEST_TIMEOUT` | `1000` 秒 | frontend 到 worker 的读超时；聚合式流式请求也使用它作为整体 deadline |
| `INFINILM_CLUSTER_BOOTSTRAP_TIMEOUT_SECONDS` | `3` 秒 | PD prefill bootstrap `/query` 的专用总超时；失败 worker 按 `max(3 * health interval, bootstrap timeout)` 退避重试 |
| `INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS` | `1` 秒 | frontend watcher 的 registry snapshot 轮询间隔；实际运行下限为 `0.1` 秒 |
| `INFINILM_CLUSTER_REGISTRY_TTL_SECONDS` | `10` 秒 | worker 注册租约时长；frontend 和 worker 应使用相同配置 |
| `INFINILM_CLUSTER_HEARTBEAT_INTERVAL_SECONDS` | `2` 秒 | worker discovery 心跳间隔；建议小于 registry TTL 的二分之一 |

etcd 客户端是可选依赖：

```bash
python -m pip install -e '.[etcd]'
```

Python 包安装后会提供 `infinilm.frontend`、`infinilm.pd-frontend`、
`infinilm.infinilm` 和 `infinilm.cluster-launch` 入口。cluster 文档和
`aggregated/`、`disaggregated/` 下的启动脚本也会随 package data 一起打包；
源码部署仍可直接运行这些脚本。

### 聚合式 frontend

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--router-mode` | `round-robin` | `round-robin`、`least-loaded`、`session-affinity` 或 `kv-aware` |
| `--router-min-initial-workers` | `1` | frontend 启动所需最小可路由 worker 数 |
| `INFINILM_CLUSTER_AFFINITY_TTL_SECONDS` | `300` 秒 | session affinity 条目生存时间；仅 session-affinity 模式读取，`0` 表示禁用 TTL |
| `INFINILM_CLUSTER_AFFINITY_CAPACITY` | `4096` | session affinity LRU 容量；仅 session-affinity 模式读取 |
| `INFINILM_CLUSTER_KV_ROUTER_MIN_TOKENS` | `1` | kv-aware 候选 worker 的最小命中 token 数；仅 kv-aware 模式读取 |
| `--http-host` / `--http-port` | `127.0.0.1` / `18000` | OpenAI API 监听地址 |

聚合式 frontend 启动时等待可路由 worker 数达到 `--router-min-initial-workers`。DRAINING worker 仍可能出现在 discovery 中，但不会计入启动等待条件。

### PD frontend

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--router-mode` | `round-robin` | `round-robin` 或 `least-loaded` |
| `--min-prefill-workers` | `1` | 模型 ready 所需最小 prefill 数 |
| `--min-decode-workers` | `1` | 模型 ready 所需最小 decode 数 |
| `--http-host` / `--http-port` | `127.0.0.1` / `18520` | PD OpenAI API 监听地址 |

PD frontend 立即监听 HTTP 端口，client bootstrap 由后台 reconciler 完成。没有可用 P/D 模型时，进程保持 `healthy`，但 `ready=false`，推理请求返回 503。

### Worker

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--served-model-name` | 模型目录名 | 对外暴露的 OpenAI model id |
| `--advertise-host` | `127.0.0.1` | 写入 registry 的可达地址 |
| `--host` / `--port` | 引擎默认值 | worker HTTP 监听地址 |
| `--disaggregation-mode` | `aggregated` | `aggregated`、`prefill` 或 `decode` |
| `--worker-type` | `aggregated` | `--disaggregation-mode` 的可选别名，同时出现时必须一致 |
| `INFINILM_CLUSTER_DRAIN_GRACE_SECONDS` | `1` 秒 | 退出前保持 DRAINING 的宽限时间；`0` 表示不等待 |
| `--mooncake-bootstrap-host` | `127.0.0.1` | prefill 写入 registry 的 bootstrap 地址 |
| `--mooncake-bootstrap-port` | `8998` | prefill 写入 registry 的 bootstrap 端口 |

身份推导关系：

```text
aggregated -> aggregated worker
prefill    -> kv producer
decode     -> kv consumer
```

PD worker 必须提供 Mooncake KV connector 配置；当前 PD 部署要求 `tp=1`、`pp=1`。

## 3. Worker Registry

### 注册记录

每条注册记录包含：

| 字段 | 说明 |
|---|---|
| `namespace` | discovery namespace |
| `component` | aggregated 为 `infinilm`，prefill 为 `prefill`，decode 为 `backend` |
| `endpoint` | 当前固定为 `generate` |
| `instance_id` | worker 实例唯一 ID |
| `worker_type` | `aggregated`、`prefill` 或 `decode` |
| `model` | worker 本地模型路径 |
| `served_model_name` | 对外模型名 |
| `host` / `port` | worker HTTP 地址 |
| `registered_at` | 首次注册时间，仅用于观测 |
| `heartbeat_at` | 最近心跳时间 |
| `lease_expires_at` | 记录中的租约截止时间 |
| `engine_healthy` | engine 健康状态 |
| `state` | `ready` 或 `draining` |
| `metadata` | 负载、流量、能力、PD bootstrap 等扩展信息 |

聚合式 frontend 只读取 aggregated worker。PD frontend 只读取 prefill 和 decode worker，不会把某一种角色误当成完整推理实例。

### file backend

file backend 仅适合同一台机器上的 frontend 和 worker。每个 worker 一个
JSON 注册文件，写入时先创建临时文件，再原子替换为正式文件，frontend 不会
读到半写记录。跨节点部署必须使用 etcd：共享文件系统不能保证各机器时钟、
本机可达性和文件锁语义满足 registry 租约要求。

file backend 的时间字段是存活判断依据：

```text
lease_expires_at = heartbeat_at + registry_ttl
```

worker 停止心跳后，frontend 在下一次 snapshot 时删除过期注册文件。

frontend 和 worker 都会从 `INFINILM_CLUSTER_REGISTRY_TTL_SECONDS` 读取注册租约时长；部署时应保证同一 namespace 内的进程使用一致配置。

### etcd backend

etcd backend 面向跨节点部署。key 格式：

```text
{etcd-prefix}/discovery/{namespace}/{component}/{endpoint}/{instance_id}
```

value 是与 file backend 相同语义的 worker JSON。worker 注册时创建 etcd lease；heartbeat 刷新 lease 并更新 value；正常退出删除 key 并 revoke lease；进程 crash 后由 etcd 自动删除过期 key。

etcd backend 中，etcd key 是否存在是存活权威；记录内时间字段主要用于观测，不依赖各机器系统时钟做存活判断。

frontend watcher 对 file 和 etcd 都按 `INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS`
周期执行 snapshot，实际运行下限为 `0.1` 秒。

## 4. Worker 生命周期

### 注册

worker 在 engine 完成初始化并进入可服务状态后才写入 registry。因此一条记录出现时，对应 HTTP 服务至少已经完成启动流程。

### 心跳

worker 周期刷新：

```text
heartbeat_at
lease_expires_at
engine_healthy
state
metadata.load
metadata.traffic
metadata.kv_index
metadata.pd
```

frontend watcher 周期读取 registry，并基于最新记录维护可路由候选。

### DRAINING 与优雅退出

worker 收到 `SIGTERM` 或第一次 `Ctrl+C` 后：

```text
state: ready -> draining
等待 drain grace
完成存量请求并停止接收新请求
注销 registry
退出进程
```

frontend 观察到 DRAINING 后不再选择该 worker。第二次退出信号会跳过宽限等待，用于运维强制中断。

### Crash 与租约过期

优雅退出会删除注册记录；进程 crash 或被 `kill -9` 时没有注销机会。file backend 等待记录时间过期，etcd backend 等待 lease 过期。过期前 frontend 可能短暂看到该 worker，请求失败后按安全重试边界切换候选。

### 扩缩容

worker 可以独立加入或退出集群。新增 worker 在下一次 watcher 轮询中进入候选；进入 DRAINING 后，frontend 停止为其分发新请求。PD frontend 支持独立扩缩 prefill 和 decode 两组 worker。

## 5. 模型解析与 API

### 模型名规则

chat completions 请求中的 `model` 规则：

1. 缺失或为 `null`：返回 HTTP 400。
2. 为空字符串：表示未指定模型，选择第一个可路由 worker 的 `served_model_name`。
3. 为非空字符串：按 `served_model_name` 精确匹配。
4. 指定模型没有可路由 worker：聚合式服务返回 404。

frontend 会把解析后的模型名写回请求体，下游 worker 看到的是确定的 served model name。

### API

聚合式 frontend：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | frontend 存活状态和 worker 数量 |
| GET | `/v1/models` | 可用模型列表 |
| GET | `/v1/models/{model}/ready` | 指定模型是否满足最小可路由副本数 |
| POST | `/v1/chat/completions` | OpenAI-compatible chat API |
| POST | `/chat/completions` | 兼容路径 |

PD frontend：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | frontend 存活状态、P/D 数量和模型 ready 状态 |
| GET | `/v1/models` | 同时存在 P/D 的模型列表 |
| POST | `/v1/chat/completions` | OpenAI-compatible chat API |
| POST | `/chat/completions` | 兼容路径 |

两种 frontend 都支持流式与非流式响应。

### 直接访问 worker

worker 自身提供：

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /chat/completions
```

直接访问适合调试模型加载、chat template 和 engine 输出，但会绕过统一路由、重试、集中指标和监控。

producer-only prefill worker 的 OpenAI surface 不允许外部直接调用。frontend 内部请求携带：

```text
X-InfiniLM-Internal: true
```

缺少该 header 时，prefill OpenAI 接口返回 403。

## 6. 聚合式路由

### 可路由候选

候选 worker 必须满足：

```text
state == ready
engine_healthy == true
served_model_name 匹配请求模型
```

DRAINING、engine 不健康或模型不匹配的 worker 不会被路由。frontend 与 worker 之间固定使用 HTTP。

### round-robin

按稳定顺序轮询候选 worker，适合作为对照基线和请求计数均匀的场景。

### least-loaded

综合负载评分：

```text
0.90 * frontend inflight
+ 0.05 * waiting_requests
+ 0.03 * running_requests / max_batch_size
+ 0.02 * kv_usage_ratio
```

选择评分最低者，平局时轮询。frontend 本地 inflight 权重最高，因为它比 worker 周期上报的负载快照更接近当前入口压力。

### session-affinity

优先复用相同会话所在的 worker，以提高本地 prefix cache 命中率。routing key 依次来自：

1. `X-InfiniLM-Session-ID`
2. `X-Session-ID`
3. 请求内部 routing key
4. OpenAI `user`
5. 模型名、首个 system message 和首个 user message 的稳定前缀指纹

key 只保存 SHA-256 指纹，不保存原始用户内容。条目具有 TTL 和容量上限。preferred worker 过载超过 slack 时，回退 least-loaded 并重新绑定会话。

### kv-aware

kv-aware 根据 worker 已发布的 prefix cache 索引选择最可能复用 KV 的 worker：

```text
frontend tokenize prompt
  -> 计算连续 prefix hash
  -> 查询每个 worker 的 cache index
  -> 过滤命中 token 数不足的候选
  -> 过滤负载超出 slack 的候选
  -> 选择命中 token 数最多者
  -> 无有效候选时回退 least-loaded
```

worker 导出 cache snapshot 和 append-only event。frontend 读取 snapshot 并 tail event，worker 退出或 DRAINING 时删除对应索引。

跨节点 etcd 部署下，frontend 无法直接读取 worker 本机 cache index 文件，因此聚合式 `etcd + kv-aware` 必须配置所有节点可访问的 `--cache-index-dir`。PD frontend 不读取这些索引，不受该限制影响。

### 重试边界

在尚未输出任何 token 前，连接错误、超时或 HTTP 5xx 可以切换下一个候选 worker。一旦已经向客户端输出内容，不再切换 worker，避免客户端收到重复生成。

## 7. PD 请求链路

一次 chat 请求的处理流程：

```text
PD frontend
  -> 从 routable pool 原子选择 prefill 和 decode client
  -> decode 请求先启动，携带 do_remote_prefill=true 并查询 bootstrap
  -> prefill 请求发送 do_remote_decode=true、max_tokens=1
  -> decode 的首个 chunk 先被 frontend 缓存
  -> prefill 成功后才向客户端输出 decode chunk
  -> 首 token 前 prefill/decode 失败则取消对端并返回真实 HTTP 错误状态
  -> 流式响应已开始后失败则取消对端并追加 SSE 错误事件
  -> 客户端断开则同时取消 prefill 和 decode
  -> PD frontend 返回 JSON 或 SSE
```

PD worker 注册的能力：

```text
prefill: internal-prefill
decode:  openai-chat, internal-decode
```

PD frontend 只选择具备对应能力的 worker，并要求同一 `served_model_name` 同时存在 prefill 和 decode。多 prefill 时每个实例使用独立 bootstrap 端口；frontend 从 registry metadata 读取 bootstrap 信息并查询 engine ID。

## 8. 健康与观测

### `/health`

聚合式 frontend 返回：

```json
{
  "status": "healthy",
  "ready": true,
  "workers": 3,
  "ready_workers": 3
}
```

字段含义：

- `workers`：discovery 可见 worker 数，包含 DRAINING。
- `ready_workers`：可路由 worker 数，只包含 ready 的 worker。
- `ready`：`ready_workers >= router_min_initial_workers`。
- `status=healthy` 只表示 frontend 进程存活，不表示一定可服务。

PD frontend 额外返回：

```json
{
  "prefill_workers": 2,
  "decode_workers": 2,
  "models": ["Qwen3-0.6B"]
}
```

### `/metrics`

指标分为：

- `infinilm_router_*`：经过 frontend router 的请求、重试、延迟、usage、路由命中和 worker 负载。
- `infinilm_worker_*`：worker HTTP 进程看到的请求、成功、失败、取消、active 和 cache 指标。
- `infinilm_pd_*`：PD frontend 的 P/D 数量、角色请求数和 per-worker inflight。

常用指标包括：

```text
infinilm_router_mode
infinilm_router_requests_total
infinilm_router_attempts_total
infinilm_router_errors_total
infinilm_router_retries_total
infinilm_router_inflight
infinilm_router_affinity_requests_total
infinilm_router_affinity_hits_total
infinilm_router_kv_aware_requests_total
infinilm_router_kv_hit_requests_total
infinilm_router_kv_predicted_saved_tokens_total
infinilm_router_kv_actual_saved_tokens_total
infinilm_worker_received_requests_total
infinilm_worker_active_requests
infinilm_pd_workers
infinilm_pd_frontend_requests_total
infinilm_pd_frontend_inflight_requests
```

direct / untracked 请求按以下差值估算：

```text
max(0, worker_received_requests_total - router_worker_requests_total)
```

它表示绕过 frontend 直接访问 worker 的流量估计。

### `/monitoring`

聚合式 frontend 默认地址：

```text
http://127.0.0.1:18000/monitoring
```

PD frontend 默认地址：

```text
http://127.0.0.1:18520/monitoring
```

页面没有独立进程或端口，随 frontend 生命周期启动和退出。页面每 2 秒请求一次 JSON 快照，并支持手动刷新和自动刷新。

聚合式页面展示 router、affinity、KV-aware、延迟分位数和 worker 负载。PD 页面按 prefill / decode 分组展示 worker、engine ID、状态、load 和 frontend inflight。

如果 frontend 监听 `0.0.0.0`，监控页面会暴露到对应网络。生产环境需要加访问控制，或仅监听本机地址。

## 9. 部署

### 聚合式集群

默认启动 1 个 frontend 和 3 个 aggregated worker，router 为 `kv-aware`，discovery 为 file：

```bash
cd /workspace_codex/InfiniLM
bash python/infinilm/cluster/aggregated/run_cluster.sh
```

默认值：

```text
frontend: 18000
workers:  18091、18092、18093
GPUs:     0、1、2
router:   kv-aware
```

常用覆盖：

```bash
# 单 worker round-robin
GPUS=0 ROUTER_MODE=round-robin \
  bash python/infinilm/cluster/aggregated/run_cluster.sh

# 三 worker least-loaded
ROUTER_MODE=least-loaded \
  bash python/infinilm/cluster/aggregated/run_cluster.sh
```

启动后访问：

```text
API:       http://127.0.0.1:18000/v1
Monitoring: http://127.0.0.1:18000/monitoring
Metrics:   http://127.0.0.1:18000/metrics
```

### PD 集群

一体化 PD 启动器默认启动 1 个 PD frontend、1 个 prefill worker 和 1 个 decode worker：

```bash
cd /workspace_codex/InfiniLM
bash python/infinilm/cluster/disaggregated/run_pd_discovery_cluster.sh
```

默认值：

```text
frontend:      18520
prefill:       18521，GPU 4
decode:        18531，GPU 5
bootstrap:     18528
router:        round-robin
KV transfer:   Mooncake TCP
```

启动 2P+2D：

```bash
PREFILL_GPUS="2 3" DECODE_GPUS="4 5" \
  bash python/infinilm/cluster/disaggregated/run_pd_discovery_cluster.sh
```

`PLAN_ONLY=1` 只校验依赖、模型路径、端口占用和启动计划，不启动服务。

### 独立 PD frontend 和 worker

frontend：

```bash
NAMESPACE=infinilm-pd \
RUN_DIR=/tmp/infinilm-pd \
PD_FRONTEND_PORT=18520 \
bash python/infinilm/cluster/disaggregated/run_pd_frontend.sh
```

prefill：

```bash
NAMESPACE=infinilm-pd \
RUN_DIR=/tmp/infinilm-pd \
GPU=4 PORT=18541 BOOTSTRAP_PORT=18548 \
ENGINE_ID=infinilm-pd-prefill-0 \
bash python/infinilm/cluster/disaggregated/run_pd_worker.sh prefill
```

decode：

```bash
NAMESPACE=infinilm-pd \
RUN_DIR=/tmp/infinilm-pd \
GPU=5 PORT=18551 \
ENGINE_ID=infinilm-pd-decode-0 \
bash python/infinilm/cluster/disaggregated/run_pd_worker.sh decode
```

扩容时再次运行 worker 入口，并保证同角色副本使用不同 GPU、HTTP 端口、bootstrap 端口和 engine ID。

### 使用 etcd

frontend 和 worker 必须使用相同 namespace、etcd endpoints 和 etcd prefix：

```bash
DISCOVERY_BACKEND=etcd \
ETCD_ENDPOINTS=127.0.0.1:2379 \
ETCD_PREFIX=/infinilm \
  bash python/infinilm/cluster/disaggregated/run_pd_frontend.sh
```

```bash
DISCOVERY_BACKEND=etcd \
ETCD_ENDPOINTS=127.0.0.1:2379 \
ETCD_PREFIX=/infinilm \
  bash python/infinilm/cluster/disaggregated/run_pd_worker.sh prefill
```

跨节点聚合式 `kv-aware` 还需要：

```bash
CACHE_INDEX_DIR=/shared/infinilm/cache-index
```

### 请求示例

非流式：

```bash
curl http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "请输出 READY"}
    ],
    "max_tokens": 32,
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
      {"role": "user", "content": "请输出 READY"}
    ],
    "max_tokens": 32,
    "temperature": 0,
    "stream": true
  }'
```

## 10. 安全与边界

- file backend 仅适合同机部署；跨节点 discovery 必须使用 etcd backend。
- worker、etcd 和 Mooncake bootstrap 不应在未加访问控制的情况下暴露到公网。
- prefill OpenAI surface 需要 internal header，防止绕过 PD frontend 直接调用 producer。
- readiness 基于 registry 状态和租约，不主动探测每个 worker 的 `/health` endpoint。
- 聚合式服务不跨 worker 迁移 KV；PD 服务通过 Mooncake 传输 prefill KV。
- 集群层不聚合多个 worker 的 batch，各 worker 内部独立调度。
- 当前 PD 部署要求 `tp=1`、`pp=1`。
- 多模型通过 `served_model_name` 区分。

## 11. 测试

运行集群相关单元测试：

```bash
cd /workspace_codex/InfiniLM
PYTHONPATH=python python -m pytest -q \
  test/service/test_cluster.py \
  test/service/test_cluster_environment.py \
  test/service/test_cluster_launch.py \
  test/service/test_routing_round_robin.py \
  test/service/test_routing_least_loaded.py \
  test/service/test_routing_session_affinity.py \
  test/service/test_routing_kv_aware.py \
  test/service/test_pd_cluster.py \
  test/service/test_pd_client_reconciler.py \
  test/service/test_pd_http_transport.py \
  test/service/test_pd_router.py \
  test/service/test_pd_worker_client_pool.py \
  test/service/test_etcd_registry.py
```

测试覆盖：

- aggregated worker 注册、心跳、租约过期和 DRAINING；
- round-robin、least-loaded、session-affinity 和 kv-aware；
- HTTP worker 非流式与流式转发；
- PD 后台 client reconcile、bootstrap 专用 timeout、并发初始化和原子 acquire；
- 首 token 前重试和 worker 5xx 切换；
- prefix hash、cache index 和实际命中指标；
- PD 角色发现、bootstrap metadata、能力过滤、真实本地 HTTP P/D 链路和 PD monitoring；
- etcd lease、注册、心跳、注销、异常记录过滤和过期删除。

多数单元测试使用 mock transport，不加载模型、不占用 GPU。`test_pd_client_reconciler.py`
中的集成测试会真实占用本机随机端口启动 bootstrap、prefill、decode 和 frontend，但不加载模型、不占用 GPU；真实多副本和 Mooncake KV 链路仍需要另行验证。
