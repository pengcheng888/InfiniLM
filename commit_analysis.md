# InfiniLM cluster 提交演进分析

分析范围从 `927f4260faf2859909d34d21b8ad52873171df2a` 开始，到当前分支
`pd-20260907` 的 HEAD `91c85738bcae8ba9fe241c73842863452e3f9c86` 为止。时间按
Asia/Shanghai 展示；每个提交的标题保持原 commit message。

| 时间 | Commit | 主题 |
|---|---|---|
| 2026-10-12 18:43:45 | `927f4260` | cluster |
| 2026-10-13 17:28:06 | `634b29c8` | least-loaded and monitoring |
| 2026-10-13 18:21:56 | `123afdd4` | DRAINING and worker_traffic |
| 2026-10-13 19:19:22 | `27ad7722` | session-affinity |
| 2026-10-14 13:55:22 | `88fdfee3` | kv-aware |
| 2026-10-14 16:31:11 | `2936ebfc` | pd-frontend |
| 2026-10-14 17:33:06 | `91c85738` | etcd发现 |

整体演进主线是：

```text
统一聚合式服务入口
  -> 负载感知与观测
  -> 优雅下线与 worker 侧观测
  -> 会话亲和
  -> prefix cache 感知路由
  -> P/D 分离 frontend 与独立生命周期
  -> 分布式 etcd discovery
```
## cluster

Commit: `927f4260faf2859909d34d21b8ad52873171df2a`
时间：2026-10-12 18:43:45 +08:00

### 解决的问题

这是 cluster 能力的基础提交。它解决的是单机多副本聚合式推理服务缺少统一入口、
服务发现和统一路由面的问题。此前单独启动的 `InferenceServer` 虽然能提供
OpenAI-compatible API，但客户端需要自己知道每个 worker 的地址，也无法在多个
完整推理实例之间做统一转发、健康过滤和故障切换。

### 新增技术

1. **`infinilm.cluster` 包**

   新增了完整的 cluster 层：

   ```text
   client
     -> ClusterFrontend
     -> RoundRobinRouter
     -> HttpWorkerClientFactory / HttpWorkerClient
     -> ClusterInferenceServer
     -> AsyncLLMEngine
   ```

2. **file discovery 控制面**

   `FileServiceRegistry` 为每个 worker 写一条 JSON 记录：

   ```text
   <namespace>.<component>.<endpoint>.<instance_id>.json
   ```

   记录包含：

   - model 与 served model name；
   - worker HTTP host/port；
   - engine health；
   - heartbeat 与 lease 过期时间；
   - transport 和 worker metadata。

   写入使用临时文件加 `os.replace()`，避免 frontend 读到半写 JSON。

3. **heartbeat / lease 生命周期**

   `WorkerDiscoveryService` 在 engine ready 后注册 worker，并周期 heartbeat。
   frontend 的 `FileServiceWatcher` 周期 snapshot discovery 目录，过滤不健康或
   lease 过期记录。

4. **统一 OpenAI frontend**

   `ClusterFrontend` 对外暴露：

   - `/v1/models`
   - `/v1/chat/completions`
   - `/chat/completions`
   - `/health`
   - `/metrics`

   frontend 将 OpenAI 请求转换为内部 frame 流，兼容流式和非流式响应。

5. **round-robin 路由与 HTTP 数据面**

   `RoundRobinRouter` 从 watcher 获取 ready worker，按顺序轮询选择实例。
   `HttpWorkerClientFactory` 复用一个共享 `httpx.AsyncClient`，避免每个请求
   都创建连接池。

6. **首 token 前安全重试**

   worker 连接失败、超时或 HTTP 5xx 时，在尚未向客户端输出内容前可以尝试下一
   个 worker；已经输出 token 后不再切换，避免客户端收到重复生成内容。

7. **基础指标**

   `metrics.py` 记录请求数、成功数、失败数、inflight 等，并以 Prometheus 文本
   格式输出。

### 关键模块

- `python/infinilm/cluster/config.py`
- `python/infinilm/cluster/registry.py`
- `python/infinilm/cluster/frontend.py`
- `python/infinilm/cluster/router.py`
- `python/infinilm/cluster/worker.py`
- `python/infinilm/cluster/worker_discovery.py`
- `python/infinilm/cluster/http_worker_client_factory.py`
- `test/service/test_cluster.py`

### 测试与验证

新增 `test/service/test_cluster.py`，覆盖：

- file registry 原子写入、heartbeat、过期删除；
- watcher 过滤 worker；
- frontend 配置解析；
- round-robin 分布；
- HTTP worker 非流式转换；
- HTTP worker 流式转换；
- worker 5xx 后切换下一副本。

### 阶段结论

该提交建立了“控制面与数据面分离”的基础结构：discovery 只承担服务元数据，
推理流量仍走 worker HTTP API。后续所有路由、监控、PD 和 etcd 能力都在这条
边界上演进。

## least-loaded and monitoring

Commit: `634b29c83091df612a112972057c2485918ac02b`
时间：2026-10-13 17:28:06 +08:00

### 解决的问题

round-robin 只能保证请求数量近似均匀，不能反映 worker 的真实排队、执行和 KV
cache 压力。长请求、模型加载后容量差异或个别 worker 慢时，轮询会把新请求继续
发给已经繁忙的实例。同时，frontend 缺少可运维的集群状态视图和延迟分布指标。

### 新增技术

1. **`ClusterRouter` 统一路由框架**

   原 `RoundRobinRouter` 演进为 `ClusterRouter`，同一 Router 内支持：

   - `round-robin`
   - `least-loaded`

   默认仍为 `round-robin`，`least-loaded` 通过 `--router-mode` 显式选择。

2. **worker 负载快照**

   `AsyncLLMEngine.get_load_stats()` 提供 best-effort 快照：

   ```text
   waiting_requests
   running_requests
   max_batch_size
   kv_capacity
   kv_used
   kv_usage_ratio
   num_free_blocks
   ```

   这些值来自 scheduler 近似状态，用作路由提示，不作为强一致计费状态。

3. **discovery 携带负载**

   `WorkerDiscoveryService` 在注册和 heartbeat 时把 `get_load_stats()` 写入
   `WorkerRecord.metadata.load`。frontend watcher 每次刷新后能看到最新快照。

4. **复合负载评分**

   `worker_load()` 解析负载字段，`load_score()` 计算：

   ```text
   score =
     0.90 * frontend inflight
   + 0.05 * waiting_requests
   + 0.03 * running_requests / max_batch_size
   + 0.02 * kv_usage_ratio
   ```

   frontend 本地 inflight 权重最高，因为它是当前入口最新、最直接的压力信号；
   worker metadata 是周期性近似值。平分时使用 cursor 打破平局，避免同一候选
   被反复选择。

5. **指标系统增强**

   `metrics.py` 扩展为内存 Prometheus 指标系统，包括：

   - request / attempt / success / error；
   - streaming 与 non-streaming；
   - per-worker requests、errors、retries；
   - per-worker duration、TTFT、inter-token latency；
   - usage prompt/completion/total tokens；
   - bucket histogram 与 p50/p95/p99。

6. **内置监控页面**

   新增 `cluster/monitoring/`：

   - `page.py`：无外部依赖的 HTML/CSS/JS 页面；
   - `snapshot.py`：聚合 router metrics 与 worker metadata；
   - `/monitoring`：页面；
   - `/api/monitoring`：页面使用的 JSON 快照。

   页面展示 worker endpoint、GPU、load、inflight、错误、延迟分位数和请求计数。

### 主要流程

```text
worker scheduler
  -> AsyncLLMEngine.get_load_stats()
  -> WorkerDiscoveryService heartbeat
  -> WorkerRecord.metadata.load
  -> FileServiceWatcher
  -> ClusterRouter.load_score()
  -> least-loaded worker
```

### 测试与验证

新增和扩展测试覆盖：

- `least-loaded` 配置解析与默认 round-robin；
- round-robin 均匀选择；
- inflight 在负载评分中占主导；
- worker metadata 缺失或非法时的降级；
- engine load stats 读取；
- `/monitoring` 与 `/api/monitoring`；
- Prometheus parser 能解析新增指标。

### 阶段结论

该提交把集群从“简单轮询”升级为“可观测 + 可按近似负载调度”，并建立了后续
session-affinity 与 kv-aware 共用的指标和 worker 状态基础设施。

## DRAINING and worker_traffic

Commit: `123afdd4d628ac6bbe6b3065b35b4f0a4ef0d6be`
时间：2026-10-13 18:21:56 +08:00

### 解决的问题

原 worker 收到 `SIGTERM` 或 `Ctrl+C` 后立即进入 Uvicorn shutdown。frontend
watcher 可能还没来得及观察到进程即将退出，仍把新请求路由过去，造成连接失败。
另外，frontend router 指标只能看到经过 frontend 的流量，直接访问 worker 的
请求不可见，导致监控低估 worker 实际压力。

### 新增技术

1. **`WorkerRecord.state`**

   discovery 记录新增状态：

   ```text
   ready     可接收新请求
   draining  存量请求可继续，新请求不再选择
   ```

2. **Router 过滤 DRAINING**

   `ClusterRouter.available_workers()` 与 `select()` 都会过滤非 `ready` 状态。
   因此 watcher 看到 `draining` 后，新请求会避让该 worker。

3. **`DrainingUvicornServer`**

   worker 收到第一次退出信号后：

   ```text
   SIGTERM / SIGINT
     -> discovery state = draining
     -> 等待 drain-grace-seconds
     -> uvicorn.should_exit = true
     -> unregister
   ```

   第二次退出信号会跳过宽限等待，保留运维强制中断手段。

4. **drain grace 配置**

   新增 `--drain-grace-seconds`。宽限期用于覆盖 frontend watcher 轮询间隔和
   在途路由决策，让 discovery 有时间收敛。

5. **worker 侧流量指标**

   新增 `WorkerTrafficMetrics`，在 worker HTTP 进程内统计：

   - received total；
   - streaming / non-streaming；
   - success / failed / canceled；
   - active requests。

   它与 frontend router 指标分离，能看到 frontend 路由流量和直接访问 worker 的
   流量。

6. **非流式与流式请求生命周期计数**

   `ClusterInferenceServer` 覆写 `_chat()` 与 `_stream_chat()`，在请求开始、
   完成、失败和取消时更新 worker traffic，并把快照写入 discovery metadata。

### 主要流程

```text
worker receives SIGTERM
  -> DrainingUvicornServer
  -> WorkerDiscoveryService.start_draining()
  -> discovery state=draining
  -> frontend watcher refresh
  -> router excludes worker
  -> grace period
  -> uvicorn shutdown
  -> unregister
```

### 测试与验证

新增测试覆盖：

- drain grace 参数；
- Uvicorn 第一次信号进入 drain、第二次信号强制退出；
- router 排除 draining worker；
- worker traffic 成功、失败、取消和 active 计数；
- 非流式与流式 chat traffic。

### 阶段结论

该提交补上了优雅下线协议，使 worker 从“直接消失”变成“先声明不可调度，再
退出”，并把观测视角从 frontend 扩展到 worker 自身。

## session-affinity

Commit: `27ad7722e816c1dfcf646353e61e508b4ff9adec`
时间：2026-10-13 19:19:22 +08:00

### 解决的问题

多轮对话或同一用户的请求如果每次落到不同 worker，本地 prefix cache 很难被
连续复用。least-loaded 只看压力，不知道请求之间可能存在稳定上下文关系；需要
一种有界、可过期、隐私安全的会话亲和机制。

### 新增技术

1. **隐私保护 affinity key**

   `affinity_key()` 按优先级生成 SHA-256 key：

   ```text
   X-InfiniLM-Session-ID
   X-Session-ID
   payload.__infinilm_routing_key__
   payload.user
   model + 首个 system message + 首个 user message 的稳定前缀
   ```

   只保存和传输哈希，不保存原始 session/user 内容。

2. **TTL + LRU AffinityStore**

   `AffinityStore` 提供：

   - key -> worker id 映射；
   - TTL 过期；
   - 容量上限；
   - LRU 淘汰；
   - worker 消失时删除相关映射；
   - 只保留当前可用 worker 的映射。

   避免 frontend 长期运行时 affinity 表无限增长。

3. **session-affinity 路由模式**

   新增 `--router-mode session-affinity`，配置项包括：

   - `--router-affinity-ttl-secs`
   - `--router-affinity-capacity`

4. **亲和与负载折中**

   如果 preferred worker 仍在候选集中，并且负载分数不超过当前最小值 `+1.0`，
   则复用该 worker；否则回退 least-loaded 并重新绑定 affinity。

   这个 slack 避免为了缓存局部性把请求持续压到明显过载的实例上。

5. **内部 routing key 不透出**

   frontend 从 header 生成或读取 routing key，并在转发前删除内部字段，避免
   worker 或下游客户端看到实现细节。

6. **affinity 指标**

   记录：

   - keyed / unkeyed requests；
   - hits / misses；
   - expired；
   - evictions；
   - active entries。

### 主要流程

```text
request
  -> affinity_key()
  -> AffinityStore.get()
  -> preferred worker ready and load within slack?
       yes -> reuse worker
       no  -> least-loaded fallback and reassign
```

### 测试与验证

新增测试覆盖：

- TTL 过期和 LRU 淘汰；
- identity 与稳定对话前缀 key；
- CLI 默认值与覆盖；
- preferred worker 复用；
- worker 不可用或过载时回退；
- frontend header 亲和和内部字段隐藏。

### 阶段结论

该提交引入了面向多轮对话的软亲和，使路由可以在“缓存局部性”和“负载均衡”
之间做有界折中。

## kv-aware

Commit: `88fdfee37889fdf79f5fd6491788fe7748ef5f7c`
时间：2026-10-14 13:55:22 +08:00

### 解决的问题

session-affinity 只能利用会话或稳定前缀的间接信号，不知道每个 worker 实际
已经缓存了哪些 token block。对于多个客户端共享长上下文、系统提示词或文档
前缀的场景，需要直接根据 worker 本地 prefix cache 状态选择最可能命中缓存的
worker，减少重复 prefill。

### 新增技术

1. **worker 侧 prefix cache 索引导出**

   新增 `cache_index.py`：

   - 观察 `BlockManager` 的 prefix index 变化；
   - 导出权威 snapshot：`prefix-cache.snapshot.json`；
   - 追加事件日志：`prefix-cache.events.jsonl`；
   - 事件类型包括 published、touched、evicted；
   - 后台线程写入，避免阻塞 scheduler；
   - 事件达到阈值后重写 snapshot 并压缩事件日志。

2. **BlockManager 观察者机制**

   `llm/cache_manager.py` 增加 index observer：

   - publish computed blocks；
   - touch 复用 blocks；
   - evict blocks；
   - observer 异常只记录，不影响推理路径。

3. **frontend 侧索引读取**

   新增 `kv_index.py`：

   - 读取每个 worker 的 snapshot；
   - tail event 文件；
   - 事件文件变短则重载 snapshot；
   - worker 退出或 DRAINING 时删除索引；
   - 维护 worker_id -> prefix hash set。

4. **请求 prefix hash 计算**

   新增 `kv_prefix.py`：

   - 加载与 worker 相同的 processor；
   - 对 chat prompt 做 template 和 tokenize；
   - 按 worker block size 计算 chained prefix hash；
   - 与 worker 本地 BlockManager 使用相同哈希算法和命名空间。

5. **kv-aware 路由模式**

   新增 `--router-mode kv-aware` 与 `--kv-router-min-tokens`。

   路由流程：

   ```text
   frontend tokenize prompt
     -> calculate chained prefix hashes
     -> KvIndexStore lookup workers
     -> 过滤 matched_tokens >= threshold
     -> 过滤负载不超过 min(load)+1.0
     -> 选择 matched_tokens 最大者
     -> 无有效候选则回退 least-loaded
   ```

6. **内部 KV prefix 字段隔离**

   frontend 将预测写入 `__infinilm_kv_prefix__`，Router 读取后会在转发前删除，
   worker 不感知该内部字段。

7. **实际命中观测**

   `InferenceServer` 响应新增 `cache` 字段：

   ```json
   {
     "hit": true,
     "matched_tokens": 512,
     "prefill_tokens": 128,
     "prompt_tokens": 640
   }
   ```

   Router 同时记录：

   - kv-aware 请求数；
   - predicted hit / miss；
   - predicted saved tokens；
   - actual saved tokens；
   - index entries。

### 设计边界

- kv-aware 是路由提示，不改变 engine 正确性；
- prefix cache 索引允许短暂滞后；
- 如果索引缺失或哈希无法计算，回退负载策略；
- 该提交阶段索引文件仍依赖 file discovery 的本机共享目录语义。

### 测试与验证

新增测试覆盖：

- BlockManager prefix lookup 和 observer event；
- worker cache index snapshot / event 导出；
- frontend 索引读取与事件回放；
- tokenizer batch encoding 提取；
- chat prompt 哈希算法一致性；
- kv-aware 选择最长命中前缀；
- internal payload 隐藏；
- Prometheus parser 接受 KV 指标。

### 阶段结论

该提交建立了“预测式 prefix cache 路由”闭环：worker 发布缓存索引，frontend
计算请求前缀哈希，Router 选择最可能节省 prefill 的 worker，并通过实际 cache
命中校准观测。

## pd-frontend

Commit: `2936ebfc56d4c4e387f1d40c08ddb182d10c62ce`
时间：2026-10-14 16:31:11 +08:00

### 解决的问题

此前 `ClusterFrontend` 面向的是聚合式 worker：每个候选都能独立完成 prefill
和 decode。PD 分离后，一次请求必须选择一个 prefill worker 和一个 decode
worker，并且 decode 需要 Mooncake bootstrap 与 engine id 才能拉取 KV。聚合式
Router 不能直接复用。该提交解决 PD 角色发现、身份收敛、独立 frontend、独立
worker 生命周期和 PD 专用观测问题。

### 新增技术

1. **角色化 file discovery**

   `FileServiceWatcher` 支持按 `worker_type` 过滤，PD frontend 只读取：

   ```text
   prefill
   decode
   ```

   聚合式 frontend 默认仍只读取 `aggregated`，两者不会混用。

2. **`--disaggregation-mode` 身份权威**

   worker 新增顶层身份配置：

   ```text
   aggregated -> component=infinilm
   prefill    -> component=prefill -> kv_role=kv_producer
   decode     -> component=backend -> kv_role=kv_consumer
   ```

   `--worker-type` 保留为兼容别名；若两者同时显式传入且不一致则报错。
   `--kv-transfer-config` 可省略 `kv_role`，由 mode 推导；显式错误 role 会被
   拒绝。

3. **PD metadata 与 capabilities**

   Worker discovery metadata 新增：

   - `pd.engine_id`
   - `pd.kv_role`
   - `pd.bootstrap_host`
   - `pd.bootstrap_port`
   - `capabilities`
   - `needs`

   典型能力：

   ```text
   prefill: internal-prefill, needs decode
   decode:  openai-chat, internal-decode, needs prefill
   ```

4. **`PdClusterFrontend`**

   新增 PD 专用 frontend，核心职责：

   - 等待并发现同 model 的 P/D；
   - 查询 prefill bootstrap `/query` 获取 engine id；
   - 维护 P/D HTTP client；
   - P 和 D 分别 round-robin 或 least-loaded；
   - 发送 prefill 请求：`do_remote_decode=true`、`max_tokens=1`；
   - 发送 decode 请求：`do_remote_prefill=true`、remote bootstrap 和 engine id；
   - 透传 JSON 或流式输出。

5. **frontend 存活与模型 ready 分离**

   PD frontend 启动时立即监听 HTTP：

   ```json
   {
     "status": "healthy",
     "ready": false
   }
   ```

   没有 P/D 时 OpenAI 请求返回 503，frontend 进程不退出。P/D 动态加入后自动
   ready。

6. **多 P / 多 D 支持**

   每个 prefill 拥有独立 HTTP port 和 bootstrap port；frontend 根据 discovery
   metadata 查询对应 bootstrap。P 与 D 可以分别扩容，角色内部独立轮询或按负载
   选择。

7. **独立生命周期脚本**

   新增：

   - `run_pd_frontend.sh`：单独启动 PD frontend；
   - `run_pd_worker.sh`：单独启动 prefill 或 decode worker；
   - `run_pd_discovery_cluster.sh`：一体化本机测试 launcher，支持多 P/D 和
     启动前端口检查；
   - `run_pd_cluster.sh`：保留静态 Mooncake proxy 对照路径。

   独立 worker 默认 3 秒 drain grace，支持 P/D 单独注册、退出和扩缩容。

8. **prefill API 安全边界**

   producer-only prefill worker 的 OpenAI API 需要内部 header：

   ```text
   X-InfiniLM-Internal: true
   ```

   外部直接访问返回 403；Mooncake proxy 和 PD frontend 内部请求会携带该 header。

9. **PD 监控页面**

   新增 `monitoring/pd_page.py`：

   - `/monitoring`：P/D 分组页面；
   - `/api/monitoring`：角色、endpoint、GPU、engine id、状态、load、inflight；
   - `/metrics`：P/D worker 数、角色请求计数和 per-worker inflight。

### 主要请求流程

```text
client
  -> PdClusterFrontend
  -> select prefill
  -> prefill worker computes prompt and publishes KV
  -> select decode
  -> decode queries prefill bootstrap
  -> Mooncake transfers KV
  -> decode streams output
  -> client
```

### 测试与验证

新增 `test/service/test_pd_cluster.py`，覆盖：

- discovery role 过滤；
- PD metadata 导出；
- P/D model 匹配与负载选择；
- round-robin 角色内轮询；
- CLI 身份解析；
- `--disaggregation-mode` 推导与校验；
- prefill OpenAI surface 403；
- 空 worker 时 frontend healthy 但 not ready；
- capability 缺失时过滤 worker；
- PD monitoring API。

### 阶段结论

该提交将 cluster 从“聚合式多副本”扩展为“PD 分离、角色化、可独立扩缩容”
的服务架构，并把 frontend 进程健康与模型可用性解耦。

## etcd发现

Commit: `91c85738bcae8ba9fe241c73842863452e3f9c86`
时间：2026-10-14 17:33:06 +08:00

### 解决的问题

file discovery 只适用于单机或共享同一文件目录的进程。跨节点部署时，frontend
和 worker 无法通过本地 `/tmp` 目录共享注册信息。该提交引入 etcd 作为分布式
discovery backend，同时保持 file 默认行为不变。

### 新增技术

1. **ServiceRegistry 协议**

   `registry.py` 抽出 backend-neutral 协议：

   ```text
   new_record
   register
   heartbeat
   unregister
   snapshot
   aclose
   ```

   `FileServiceWatcher` 不再依赖具体 `FileServiceRegistry` 类型，而是消费协议。

2. **registry factory**

   新增 `registry_factory.py`：

   ```text
   discovery-backend=file -> FileServiceRegistry
   discovery-backend=etcd -> EtcdServiceRegistry
   ```

   aggregated frontend、PD frontend 和 worker 都改为通过工厂创建 registry。

3. **`EtcdServiceRegistry`**

   etcd key 格式：

   ```text
   {prefix}/discovery/{namespace}/{component}/{endpoint}/{instance_id}
   ```

   value 仍为 `WorkerRecord` JSON，因此 Router、PD frontend 和 monitoring 不需要
   理解 etcd。

4. **etcd lease 语义**

   worker 注册流程：

   ```text
   grant lease
   -> put WorkerRecord with lease
   -> heartbeat refresh lease and update JSON
   -> normal shutdown delete key and revoke lease
   -> crash 后 lease 到期，etcd 自动删除 key
   ```

   key 是否存在是存活权威；record 中的 heartbeat 时间只用于观测，避免跨节点
   时钟偏差影响路由。

5. **heartbeat 恢复**

   etcd heartbeat 失败时，`EtcdServiceRegistry` 尝试重新申请 lease 并重写 key；
   `WorkerDiscoveryService` 捕获 heartbeat 异常并按短间隔重试，避免瞬时网络
   抖动直接终止 worker。

6. **etcd 配置**

   新增参数和环境变量：

   - `--discovery-backend etcd`
   - `--etcd-endpoints`
   - `--etcd-prefix`
   - `--etcd-username`
   - `--etcd-password-env`
   - `--etcd-ca-cert`
   - `--etcd-cert-file`
   - `--etcd-key-file`
   - `--etcd-timeout`

   对应环境变量包括：

   ```text
   DYN_DISCOVERY_BACKEND
   ETCD_ENDPOINTS
   ETCD_PREFIX
   ETCD_USERNAME
   ETCD_PASSWORD
   ETCD_CA_CERT
   ETCD_CERT_FILE
   ETCD_KEY_FILE
   ETCD_TIMEOUT
   ```

7. **optional dependency**

   `pyproject.toml` 新增：

   ```toml
   [project.optional-dependencies]
   etcd = ["etcd3>=0.12.0,<0.13"]
   ```

   安装方式：

   ```bash
   python -m pip install -e '.[etcd]'
   ```

8. **脚本接入**

   `run_pd_frontend.sh`、`run_pd_worker.sh` 和 `run_pd_discovery_cluster.sh`
   均支持：

   ```text
   DISCOVERY_BACKEND=etcd
   ETCD_ENDPOINTS=...
   ETCD_PREFIX=...
   ```

   file 仍是默认 backend。

9. **跨节点 kv-aware 约束**

   aggregated `kv-aware` 依赖 worker 本地 cache-index 文件。跨节点 etcd 场景
   下，如果没有共享文件系统，frontend 无法读取这些文件。因此：

   - 新增 `--cache-index-dir`；
   - `etcd + kv-aware` 未配置共享 cache-index 目录时启动报错；
   - PD frontend 不读取 aggregated cache-index，不受该限制影响。

### 当前实现边界

- watcher 仍按 `--health-interval` 执行 etcd prefix range 轮询；
- 尚未切换为 etcd watch 增量推送；
- etcd client 为 optional dependency；
- 多 endpoint 采取轮换尝试，focus 是发现可用 backend，而不是完整连接池治理。

### 测试与验证

新增 `test/service/test_etcd_registry.py`，覆盖：

- file / etcd 配置解析；
- registry factory；
- endpoint URL 解析；
- register、snapshot、heartbeat、unregister；
- namespace 与 worker type 过滤；
- malformed JSON 忽略；
- lease 过期后 worker 消失；
- etcd client close。

真实集成验证还包括本机 etcd：

1. 空集群时 PD frontend `healthy / ready=false`；
2. P/D 注册后 etcd key 出现，frontend ready；
3. 请求成功并触发 Mooncake KV transfer；
4. decode 优雅退出后 etcd key 删除，frontend 保持运行；
5. TTL 到期后 key 自动删除。

### 阶段结论

该提交把 discovery 从单机文件注册表抽象并扩展到分布式 etcd，同时保持
`WorkerRecord`、Router 和监控协议稳定，为多机 P/D 部署打下控制面基础。

## 总体技术演进总结

1. **控制面演进**

   ```text
   本机 JSON file discovery
     -> role-aware file discovery
     -> ServiceRegistry abstraction
     -> etcd lease-backed discovery
   ```

2. **数据面路由演进**

   ```text
   round-robin
     -> least-loaded
     -> session-affinity
     -> kv-aware prefix cache routing
     -> PD role-aware routing
   ```

3. **生命周期演进**

   ```text
   ready / lease expired
     -> ready / draining / lease expired
     -> graceful worker drain
     -> frontend 常驻、worker 独立扩缩容
   ```

4. **观测演进**

   ```text
   基础请求计数
     -> frontend 延迟与 per-worker 指标
     -> worker 侧 direct traffic
     -> prefix cache 预测/实际命中
     -> PD 角色监控
   ```

5. **推理服务形态演进**

   ```text
   多个完整 aggregated worker
     -> 负载与缓存感知 aggregated worker
     -> prefill / decode 分离
     -> frontend 与 worker 独立生命周期
     -> 支持分布式 discovery
   ```
