# Request Timing Tools

`request_timing` 是 request timing 子系统，包含运行时 recorder、日志
校验、指标计算和说明文档。该命名强调的是 timing 数据和观测能力，而不
只是一个计时器对象。

## 文件结构

```text
request_timing/
├── __init__.py           # 包公共导出
├── request_timing.py      # 运行时 JSONL recorder
├── validate_logs.py       # 检查日志完整性和数据一致性
├── compute_metrics.py     # 从日志计算普通服务和 PD 服务指标
└── README.md
```

运行时默认输出目录：

```text
/tmp/infinilm-request-timing/<pid>/<request_id>.jsonl
```

每个 worker 进程有一个 PID 目录。PD 请求会出现在 prefill 和 decode 两个
PID 目录中，但使用同一个 `request_id`，便于离线关联。

## 日志事件

### `event=step`

每一轮 scheduler/model runner/forward 处理记录一条 step。

普通 aggregated 服务：

```text
step 0: prefill
step 1..N: decode
finish
```

PD 服务：

```text
prefill worker: prefill step + finish + kv_send
decode worker:  kv_receive + decode steps + finish
```

关键字段：

```text
server_received_at_ns
  worker HTTP route 收到请求的时间，早于 body 读取、JSON 解析和 tokenize。

scheduler_enqueued_at_ns
  请求完成预处理并进入 scheduler waiting queue 的时间。

scheduler_polled_at_ns
  scheduler 从 waiting/running queue 取出请求的时间。

scheduler_batch_ready_at_ns
  scheduler 完成本轮 batch 构建的时间。

scheduler_waiting_queue_depth_before / _after
  schedule() 入口与 batch 构建完成时，waiting queue 的深度。

scheduler_running_queue_depth_before / _after
  schedule() 入口与 batch 构建完成时，running queue 的深度。

scheduler_remote_kv_requests_before / _after
  schedule() 入口与 batch 构建完成时，等待远端 KV ready 的请求数。

engine_forward_started_at_ns
  ModelRunner 调用 model_engine.forward() 前的时间。

engine_forward_completed_at_ns
  model_engine.forward() 返回后的时间。

forward_batch_id
  同一次物理 engine forward 的批次 ID。同一 batch 内所有 request step
  共享该 ID 和相同 forward start/complete 时间；离线指标按它去重，
  得到真实 engine 调用次数、batch size 分布和 forward busy ratio。
```

`durations_ns` 中的含义：

```text
queue（raw 兼容字段）
  server_received -> scheduler_polled。
  该区间混合了入口预处理和 scheduler 队列等待，离线报告不再把它作为
  独立分析指标；decode step 中该值会随请求生命周期增长。

scheduler
  scheduler_polled -> scheduler_batch_ready。

scheduler_batch_ready_to_engine_forward_start
  scheduler_batch_ready -> engine_forward_started。
  该区间覆盖 ModelRunner 入口、KV connector 前置处理和 model input 构建，
  v3 起不再单独记录 ModelRunner 入口时间。

engine_forward
  engine_forward_started -> engine_forward_completed。
```

### `event=finish`

请求进入终态时输出一条 finish 汇总，包含：

```text
steps
engine_forward_total_ns
first scheduler / engine forward 关键时间
last engine forward 完成时间
finish_reason
prompt_tokens
generated_tokens
finish_local_monotonic_ns
```

### 首次调度路径指标

离线工具会从每个 request 的首条 step 计算以下同节点连续区间。P/D 部署时
这些指标只在当前 worker 的本地 `CLOCK_MONOTONIC` 域内相减：

```text
server_to_scheduler_enqueue_ms
  server_received -> scheduler_enqueued。包含 body 读取、JSON 解析、
  chat template、tokenize 等入口处理。

scheduler_enqueue_to_polled_ms
  scheduler_enqueued -> scheduler_polled。表示进入队列后等待调度。

scheduler_batch_build_ms
  scheduler_polled -> scheduler_batch_ready。

scheduler_batch_ready_to_engine_forward_start_ms
  scheduler_batch_ready -> engine_forward_started。只取首条 step。

server_to_engine_forward_start_ms
  server_received -> engine_forward_started。它是前 4 个原子区间的总和。

first_engine_forward_ms
  engine_forward_started -> engine_forward_completed。
```

`server_to_engine_forward_start_ms + first_engine_forward_ms` 等于该 worker
本地首 token 时间。

请求级 forward 汇总还会输出 `prefill_forward_total_ms` 与
`decode_forward_total_ms`。二者都是 request 视角指标；同一物理 batch 内
多个 request 会共享同一段 forward duration，因此不能把它们跨 request 相加
当作 GPU busy 时间。节点级 GPU busy 必须按 `forward_batch_id` 去重后计算。

schema v1/v2 历史日志仍包含 `model_runner_entered_at_ns`，并拆分为
`dispatch` 与 `model_input_build`；离线校验器兼容这些日志。v3 起运行时只写
上文的合并区间。

### Scheduler 状态快照

schema v4 起，每条 step 都携带六个调度状态快照：

```text
scheduler_waiting_queue_depth_before
scheduler_running_queue_depth_before
scheduler_remote_kv_requests_before
scheduler_waiting_queue_depth_after
scheduler_running_queue_depth_after
scheduler_remote_kv_requests_after
```

`before` 在 `schedule()` 入口采样；`after` 在 deferred requests 已放回队列、
本轮 batch 即将交给 ModelRunner 时采样。同一 batch 内所有 request step 共享
同一组快照。after 不包含已被选中的 `batch_size` 个请求。

完整调度 backlog 定义为：

```text
active_before
  = waiting_before + running_before + remote_kv_before

active_after_including_scheduled
  = waiting_after + running_after + remote_kv_after + batch_size
```

这些队列深度是 scheduler/batch 级快照，不能按 request 相加。离线工具先按
`forward_batch_id` 去重，再输出每个深度的 batch-weighted 分布。

### `event=kv_send`

只在 prefill worker 上输出，统计 Mooncake producer batch transfer：

```text
started_at_ns / completed_at_ns / duration_ns
bytes
block_count
batch_size
outcome
scope=producer_batch_transfer
batch_id
batch_request_count
batch_block_count
batch_bytes
request_block_count
```

这是最接近实际 KV 网络传输耗时的指标，包含 KV 数据写入和 flag 写入。
`bytes`、`block_count` 与 `batch_bytes`、`batch_block_count` 表示整个
producer batch 的总量；同一 batch 内每个 request 都会看到相同值。
`request_block_count` 才是该 request 自身参与的 block 数。做全局传输量
统计时必须先按 `batch_id` 去重，不能把 request event 直接相加。

### `event=kv_receive`

只在 decode worker 上输出，统计 D 节点本地 KV readiness：

```text
started_at_ns / completed_at_ns / duration_ns
bytes
block_count
outcome
scope=decode_local_kv_readiness
```

该指标包含 D 等待 P 发布 KV、bootstrap 查询、ZMQ 请求、Mooncake 传输
和 flag 就绪。由于 P/D 并行启动，它通常大于 P 侧 `kv_send`，不能当作纯
网络传输时间。

## 跨机器时间约束

所有 P/D 时间均来自当前进程的 `CLOCK_MONOTONIC`。不同机器的 monotonic
epoch 不保证一致，因此 KV 事件都带有：

```json
{
  "clock_domain": "local-monotonic",
  "cross_node_comparable": false
}
```

分析规则：

```text
禁止: D timestamp - P timestamp
允许: 同一节点内 completed_at - started_at
允许: 分别统计 P duration 和 D duration
```

## 日志校验

```bash
PYTHONPATH=python python -m infinilm.llm.request_timing.validate_logs \
  --timing-root /tmp/infinilm-request-timing
```

常用参数：

```text
--request-id ID        只检查一个请求
--allow-incomplete     服务仍在运行或存在未 finish 请求时使用
--format json          输出机器可读 JSON
```

检查内容：

```text
JSON 是否可解析
request_id / role 是否一致
step_index 是否从 0 连续
phase 序列是否符合 aggregated/prefill/decode 模式
step 时间戳是否单调
durations_ns 是否与时间戳差值一致
finish 汇总是否与 step 明细一致
PD 是否同时存在 prefill/decode、kv_send/kv_receive
```

校验器兼容早期没有 `role`、`scope`、`clock_domain` 字段的日志：缺失
`role` 时按事件或 phase 推断角色；但缺失关键时间戳或 duration 不一致仍
会判为 invalid。
新 schema 还会校验同一 `forward_batch_id` 的 step 数量、batch size 和
时间戳，以及同一 `kv_send batch_id` 的 request 数、总量和复制字段是否
一致。
同时会检查 `schema_version`、`forward_batch_id` 类型、文件名与 request id
的安全映射，以及 PD 中 D 首 decode forward 是否早于最终 KV receive 完成。

退出码：

```text
0  完整且有效
1  存在 incomplete 请求
2  存在 invalid 文件或数据
```

## 指标计算

```bash
PYTHONPATH=python python -m infinilm.llm.request_timing.compute_metrics \
  --timing-root /tmp/infinilm-request-timing
```

常用参数：

```text
--request-id ID        只计算一个请求
--format json          输出机器可读 JSON
```

### 普通 aggregated 服务

```text
ttft_local_ms
  server receive -> 第一条 prefill engine forward 完成。

prefill_compute_ms
  第一条 prefill step 的 engine_forward。

decode_compute_ms
  所有 decode step 的 engine_forward 总和。
```

### PD 服务

```text
p_compute_ms
  P 节点 prefill engine_forward。

d_compute_ms
  D 节点所有 decode engine_forward 总和。

mooncake_transfer_ms
  P 节点 kv_send duration 的 request-weighted 平均，最接近单个请求
  观察到的批量传输耗时。

d_kv_readiness_ms
  D 节点 kv_receive duration，表示 D 本地 KV ready 总路径。

decode_start_after_kv_ready_ms
  D 节点 kv_receive completed -> 第一条 decode engine forward started。
```

除 request 汇总外，工具还输出节点级统计：

```text
aggregated_node / prefill_node / decode_node
  physical_forward_count
  batch_size_counts
  forward_duration_ms
  forward_busy_ratio
  process_count / processes

mooncake_transfer.kv_send
  unique_batch_count
  batch_weighted_duration_ms
  batch_bytes_total
  busy_union_ms
  throughput_during_busy_gbps
```

所有分布对象均输出：

```text
count
avg_ms
min_ms
max_ms
p50_ms
p95_ms
```

其中 p50 使用 median，p95 使用线性插值分位数。

`request_weighted_duration_ms` 按每个 request 事件平均；同一个 Mooncake
batch 内多个 request 会重复看到同一 duration。`batch_weighted_duration_ms`
先按 `batch_id` 去重再平均。`throughput_during_busy_gbps` 用去重后的
传输 bytes 除以重叠传输区间的 union 时长，因此并发 Mooncake worker 的
重叠传输会被正确合并。

`busy_union_ms` 的跨 PID 处理方式是：先在每个 PID 内合并重叠 interval，
再把各 PID 的本地 busy 时长相加。该值是本地传输 busy 时间总和，不是
跨机器全局 wall-clock interval；不同机器的 monotonic epoch 不能直接比较。

PD 汇总不计算跨节点 end-to-end latency。若需要全局时延，应在 frontend
单独记录 wall-clock 或 NTP/PTP 对齐后的时间；当前 worker timing 不做该
假设。

节点级 forward 统计也遵守上述时钟约束。`forward_duration_ms`、
`physical_forward_count`、`forward_busy_ms` 可以跨进程累加；但
`local_wall_ms` 和 `forward_busy_ratio` 只在当前 role 只有一个 PID 时输出。
如果存在多个 PID，顶层 `local_wall_ms` / `forward_busy_ratio` 为 `null`，
应查看 `processes` 列表中每个 PID 自己的 wall 时间和 busy ratio。

## 内存缓冲和强制落盘

运行时 recorder 为了减少文件 IO：

```text
首条 step 立即写出
后续 step 先保存在内存
达到 256 条未落盘 step 或距离上次写出 10 秒时强制 flush
finish 时写出剩余 step 和 finish 汇总
```

因此短请求通常只修改文件两次；长请求会按阈值分段写出。进程退出时，
recorder 会尽量 flush 未完成请求的缓冲 step。

## Recorder 统计

运行时可通过 `RequestTimingFileRecorder.stats()` 读取：

```text
dropped_records   后台队列满时丢弃的观测记录数
pending_jobs      已提交且尚未完成的后台写任务数
buffered_requests 当前仍有未落盘 step 的 request 数
```

过期文件清理在 recorder 首次写入时立即执行一次，之后每小时执行一次；
默认保留 24 小时。
