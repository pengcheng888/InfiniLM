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

## 运行模式

### 模式选择

| 模式 | 输出内容 | forward 总耗时 | 适用场景 |
|---|---|---|---|
| `full` | 每个请求每个 scheduler step 一条 `event=step`，加 finish 和 KV event | 精确总和 | 分析 batch 分布、调度快照、GPU busy union、逐步异常 |
| `summary + interval=1` | 每个 finish 一条汇总，无 step JSONL | 精确总和 | 常规低开销请求级分析 |
| `summary + interval>1` | 每个 finish 一条汇总，无 step JSONL；decode duration 采样 | 估算值 | 默认模式；适合压测和粗粒度瓶颈归因 |

summary 模式始终保留：

```text
入口分段 timestamp
scheduler enqueue / first polled / batch ready
first engine forward start / complete
last engine forward complete
decode step count 与采样元信息
PD 场景的 kv_send / kv_receive event
```

summary 模式不保留：

```text
逐 step batch size
逐 step scheduler queue snapshot
forward_batch_id
物理 forward batch 分布和节点级 forward busy ratio
```

### 运行时配置

```text
INFINILM_REQUEST_TIMING=true|false
INFINILM_REQUEST_TIMING_MODE=full|summary
INFINILM_REQUEST_TIMING_SAMPLE_INTERVAL=32
INFINILM_REQUEST_TIMING_DIR=/tmp/infinilm-request-timing
```

字段含义：

| 环境变量 | 默认值 | 合法值 | 含义 |
|---|---|---|---|
| `INFINILM_REQUEST_TIMING` | `true` | `1/true/yes/on` 或 `0/false/no/off` | 总开关。`false` 使用 no-op recorder，不采集 entrance、scheduler、finish 或 KV timing。 |
| `INFINILM_REQUEST_TIMING_MODE` | `summary` | `full/summary` | `full` 保留逐 step；`summary` 只写 finish 汇总。 |
| `INFINILM_REQUEST_TIMING_SAMPLE_INTERVAL` | `32` | `1/4/8/16/32/64/128` | summary 的 decode duration 采样间隔。仅 `1` 表示精确汇总。 |
| `INFINILM_REQUEST_TIMING_DIR` | `/tmp/infinilm-request-timing` | 目录路径 | timing root。实际文件写入 `<root>/<pid>/<request_id>.jsonl`。 |

布尔值大小写不敏感，但必须显式匹配上表集合；非法值在进程启动时抛错，避免实验配置被误判为默认值。

`sample_interval` 只影响 `summary` 模式；`full` 模式保留每个 step，因此该值
不参与 forward 总量计算。

默认配置等价于：

```bash
export INFINILM_REQUEST_TIMING=true
export INFINILM_REQUEST_TIMING_MODE=summary
export INFINILM_REQUEST_TIMING_SAMPLE_INTERVAL=32
export INFINILM_REQUEST_TIMING_DIR=/tmp/infinilm-request-timing
```

sampled summary 的 `engine_forward_total_estimate_ns` 是基于 decode 采样均值
的估算值，不是精确总和。需要精确 request 汇总时使用：

```text
INFINILM_REQUEST_TIMING_MODE=summary
INFINILM_REQUEST_TIMING_SAMPLE_INTERVAL=1
```

设置 `INFINILM_REQUEST_TIMING=false` 时，worker 不记录 entrance、
scheduler、KV transfer 或 finish timing，也不写上述目录。该模式用于性能
压测；关闭后无法使用本目录的工具做瓶颈分析。

每个 worker 进程有一个 PID 目录。PD 请求会出现在 prefill 和 decode 两个
PID 目录中，但使用同一个 `request_id`，便于离线关联。

### 主要运行时函数

| 函数 / 类 | 含义 |
|---|---|
| `RequestTimingConfig.from_environment()` | 读取并校验上述环境变量，生成 immutable 配置。 |
| `RequestTimingConfig.create_recorder()` | enabled 时创建 `RequestTimingFileRecorder`，disabled 时创建 no-op recorder。 |
| `RequestEntranceTiming` | 从 HTTP route 到 `InferenceRequest` 创建前的入口分段 timestamp 容器。 |
| `RequestStepTiming` | full 模式中一个 request 在一次 prefill/decode step 的完整观测记录。 |
| `request_timing_path()` | 把 request id 清洗为安全文件名，并定位 `<root>/<pid>/<request_id>.jsonl`。 |
| `RequestTimingFileRecorder.record_steps()` | full 模式提交一轮 scheduler batch 中各 request 的 step；首条 step 立即写出。 |
| `RequestTimingFileRecorder.record_finish()` | 请求终态时写出剩余 full step 和 finish 汇总；summary 模式只写 finish。 |
| `RequestTimingFileRecorder.record_event()` | 立即提交 `kv_send` / `kv_receive` 等非 step 事件，保持与 step/finish 同一写线程顺序。 |
| `RequestTimingFileRecorder.stats()` | 返回 dropped/pending/buffered 写入状态。 |
| `InferenceRequest.record_summary_step()` | summary 模式记录第一次 scheduler poll/batch-ready 边界，不保留 step 对象。 |
| `InferenceRequest.record_summary_prefill_forward()` | summary 模式记录精确 prefill forward。 |
| `InferenceRequest.record_summary_decode_forward()` | summary 模式统计每个 decode step，并按 interval 采样 forward duration。 |
| `InferenceRequest.get_timing_summary()` | 生成 full 或 summary finish payload 的请求级汇总字段。 |

### Schema 版本

| schema | 模式 | 关键区别 |
|---|---|---|
| v6 | `full` | 逐 step JSONL，包含 entrance duration、scheduler snapshot、`forward_batch_id`。 |
| v7 | `summary + interval=1` | 只有一条 finish；decode forward 总量精确。 |
| v8 | `summary + interval>1` | 只有一条 finish；包含采样计数和 forward 总量估算。 |

历史 v1--v5 日志仍可被校验和指标工具读取；本文中的 v6+ 字段不代表旧日志一定存在。

### 数据产生调用链

```text
HTTP chat_completions()
  -> server_received / body_read / json_parse / route_prepare

stream:
  _start_stream_chat_request()
    -> handler_started
    -> AsyncLLMEngine.start_chat_preprocessing()
       chat_template_completed
       preprocessing worker 中 multimodal/tokenization/prompt ids
  -> StreamingResponse
  -> _stream_chat()
       await finish_chat_preprocessing()

non-stream:
  _chat()
    -> handler_started
    -> AsyncLLMEngine.add_chat_request_async()
       内部同样走 preprocessing 和 request 创建

finish_chat_preprocessing() / add_request_async()
  -> _request_from_tokenized_input()
     request_created

Scheduler.add_request()
  -> scheduler_enqueued

Scheduler._note_scheduler_polled()
  -> full: 每次 poll 更新
  -> summary: 只记录第一次 poll

Scheduler._make_scheduler_output()
  -> scheduler_batch_ready
  -> full: 创建 RequestStepTiming
  -> summary: record_summary_step()

ModelRunner.executeModel()
  -> engine_forward_started
  -> model_engine.forward()
  -> engine_forward_completed
  -> full: 填充 RequestStepTiming
  -> summary: record_summary_prefill/decode_forward()

AsyncLLMEngine step loop
  -> full: recorder.record_steps()
  -> summary: 不调用 record_steps()
  -> 终态: recorder.record_finish()
  -> KV stats: recorder.record_event()
```

Mooncake worker 线程只生成本地 `kv_send` / `kv_receive` event，engine step 在
收到 `kv_connector_stats` 后统一移交给 recorder，避免多个传输线程并发写文件。

## 日志事件

### `event=step`

每一轮 scheduler/model runner/forward 处理记录一条 step。
该事件只存在于 `full` 模式；`summary` 模式不创建、不缓冲、不写出 step 记录。

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

request_created_at_ns
  InferenceRequest 完成创建的时间。schema v5 使用它拆分
  server_received -> request_created 与 request_created -> scheduler_enqueued。

body_read_completed_at_ns / json_parsed_at_ns / route_prepared_at_ns
  schema v6 入口分段点。分别表示 HTTP body 读取完成、JSON 解析完成、
  route 完成消息规范化 / request id / stream 判定。

handler_started_at_ns / preprocessing_started_at_ns
  schema v6 分别表示 `_chat()` 或 `_stream_chat()` generator 开始执行、
  开始调用 AsyncLLMEngine 预处理。
  流式请求的预处理 task 会在返回 StreamingResponse 前创建；此时
  handler_started 表示 route 开始该预处理 task，而不是等待 generator
  首次被 ASGI 迭代。

chat_template_completed_at_ns / multimodal_resolved_at_ns
  schema v6 分别表示 chat template 渲染完成、多模态输入解析完成。

tokenization_completed_at_ns / prompt_token_ids_ready_at_ns
  schema v6 分别表示 tokenizer/processor 完成、token ids 转成 Python list
  且多模态 token 映射完成。

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

server_to_request_created（v5）
  server_received -> request_created。包含 body 读取、JSON 解析、
  StreamingResponse 调度、chat template、tokenize 和 request 对象创建。

request_created_to_scheduler_enqueue（v5）
  request_created -> scheduler_enqueued。包含 output queue 初始化、
  prompt 校验、block hash 初始化和入队前处理。

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

full 模式的 finish 与 step 明细交叉校验：

```text
steps = len(step records)
engine_forward_total_ns = sum(step.engine_forward_completed - started)
first / last timestamp 与首末 step 一致
```

summary 模式的 finish 主要字段：

```text
timing_mode=summary
sample_interval
decode_step_count
sampled_decode_step_count
sampled_decode_forward_ns
prefill_forward_ns          # 内部累计，finish payload 当前不直接导出
engine_forward_total_ns     # interval=1 时输出
engine_forward_total_estimate_ns  # interval>1 时输出
```

采样规则：

```text
第 1 个 decode step 总是采样
之后 decode_step_count % sample_interval == 0 的 step 采样
```

例如 1024 个 decode step、`sample_interval=32` 时：

```text
decode_step_count=1024
sampled_decode_step_count=33
# 采样点为 1, 32, 64, ..., 1024
```

精确 summary 公式：

```text
sample_interval=1
engine_forward_total_ns
  = prefill_forward_ns + sampled_decode_forward_ns
```

估算 summary 公式：

```text
sample_interval>1
sampled_decode_average_ns
  = sampled_decode_forward_ns / sampled_decode_step_count

engine_forward_total_estimate_ns
  = prefill_forward_ns
  + sampled_decode_average_ns * decode_step_count
```

不同角色的解释：

```text
aggregated:
  prefill_forward_ns 是首段 prefill；
  decode_step_count 通常等于 generated_tokens - 1。

PD prefill:
  只执行本地 prefill forward，没有 decode 采样；
  total 只包含 P 本地 prefill。

PD decode:
  首个本地 forward 就是 decode forward；
  decode_step_count 通常等于 generated_tokens；
  total 只包含 D 本地 decode。
```

因此 `engine_forward_total_*` 是 request 视角值。full 模式中同一物理 batch 的多个 request 会复制同一段 forward duration；summary 模式没有 batch id，也无法还原物理 batch。

### 首次调度路径指标

离线工具会从每个 request 的首条 step 或 summary finish 中的等价 timestamp
计算以下同节点连续区间。P/D 部署时这些指标只在当前 worker 的本地
`CLOCK_MONOTONIC` 域内相减：

```text
server_to_scheduler_enqueue_ms
  server_received -> scheduler_enqueued。包含 body 读取、JSON 解析、
  chat template、tokenize 等入口处理。

server_to_request_created_ms
  server_received -> request_created。schema v5 起可用；它包含 HTTP body、
  JSON、stream generator 调度、chat template、tokenize 和 request 创建。

server_to_body_read（v6）
  server_received -> body_read_completed。覆盖 ASGI body 读取等待。

body_read_to_json_parse（v6）
  body_read_completed -> json_parsed。覆盖 JSON decode。

json_parse_to_route_prepare（v6）
  json_parsed -> route_prepared。覆盖消息校验、规范化、request id 和
  stream 分支判定。

route_prepare_to_handler_start（v6）
  route_prepared -> handler_started。流式请求覆盖 StreamingResponse
  generator 调度；非流式请求通常接近 0。

handler_start_to_preprocessing_start（v6）
  handler_started -> preprocessing_started。覆盖消息提取和 SamplingParams
  构建。

preprocessing_to_chat_template（v6）
  preprocessing_started -> chat_template_completed。

chat_template_to_multimodal_resolve（v6）
  chat_template_completed -> multimodal_resolved。

multimodal_to_tokenization（v6）
  multimodal_resolved -> tokenization_completed。

tokenization_to_prompt_token_ids（v6）
  tokenization_completed -> prompt_token_ids_ready。覆盖 tensor flatten、
  list 转换和多模态 token index mapping。

prompt_token_ids_to_request_created（v6）
  prompt_token_ids_ready -> request_created。覆盖 SamplingParams 补默认值、
  kv 参数复制和 InferenceRequest 对象创建。

request_created_to_scheduler_enqueue_ms
  request_created -> scheduler_enqueued。schema v5 起可用；它包含 request
  创建完成后到 scheduler waiting queue 入队前的本地处理。

scheduler_enqueue_to_polled_ms
  scheduler_enqueued -> scheduler_polled。表示进入队列后等待调度。

scheduler_batch_build_ms
  scheduler_polled -> scheduler_batch_ready。

scheduler_batch_ready_to_engine_forward_start_ms
  scheduler_batch_ready -> engine_forward_started。
  full 模式取首条 step；summary 模式取 finish 中的 first forward 边界。

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

schema v4 起，full 模式的每条 step 都携带六个调度状态快照。summary 模式没有
step 记录，因此不输出这些快照，也无法计算 batch-weighted backlog 分布：

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
mooncake_data_transfer_bypassed
```

这是最接近实际 KV 网络传输耗时的指标，包含 KV 数据写入和 flag 写入。
`bytes`、`block_count` 与 `batch_bytes`、`batch_block_count` 表示整个
producer batch 的总量；同一 batch 内每个 request 都会看到相同值。
`request_block_count` 才是该 request 自身参与的 block 数。做全局传输量
统计时必须先按 `batch_id` 去重，不能把 request event 直接相加。

`mooncake_data_transfer_bypassed` 来自诊断开关
`INFINILM_MOONCAKE_DATA_TRANSFER_BYPASS`：

```text
false:
  字段值为 false，bytes/block_count 表示真实计划并执行的数据面规模。

true:
  字段值为 true，P 侧只跳过 Mooncake engine 写调用，D 侧跳过 ready-flag
  轮询；bootstrap 和 ZMQ metadata 控制面仍执行。此时 bytes/block_count
  表示原本应传输的逻辑规模，不代表实际网络流量。
```

bypass 事件的 `bytes / duration` 不能当作真实 Mooncake 带宽。该模式只用于
性能归因，decode KV 内容无效，不能做生成正确性验证。

### `event=kv_receive`

只在 decode worker 上输出，统计 D 节点本地 KV readiness：

```text
started_at_ns / completed_at_ns / duration_ns
bytes
block_count
outcome
scope=decode_local_kv_readiness
mooncake_data_transfer_bypassed
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

## 输出路径 timing

request timing 关注入口、scheduler、forward 和 KV transfer。流式输出从 engine
线程 callback 到 ASGI send 的路径由 `python/infinilm/llm/output_timing.py`
单独观测，默认关闭。

### 配置

| 环境变量 | 默认值 | 合法值 / 范围 | 含义 |
|---|---|---|---|
| `INFINILM_OUTPUT_TIMING` | `false` | `1/true/yes/on` 视为 true，其余视为 false | 输出路径观测总开关。 |
| `INFINILM_OUTPUT_TIMING_DIR` | `/tmp/infinilm-output-timing` | 目录路径 | 输出 timing root，实际写入 `<root>/<pid>/`。 |
| `INFINILM_OUTPUT_TRACE_SAMPLE_RATE` | `0.01` | `0.0 <= rate <= 1.0` | 除首 token 外，随机保留 token timestamp trace 的比例。 |
| `INFINILM_EVENT_LOOP_LAG_INTERVAL_MS` | `10.0` | `0.5 <= interval <= 1000.0` | event loop lag monitor 的目标唤醒间隔。 |

每个请求输出：

```text
/tmp/infinilm-output-timing/<pid>/<request_id>.jsonl
event=output_timing_finish
```

服务正常 shutdown 时输出：

```text
/tmp/infinilm-output-timing/<pid>/process_summary.jsonl
event=output_timing_process_summary
```

### 请求级字段含义

| 字段 / duration | 含义 |
|---|---|
| `token_count` | 观测到的输出 token 数。 |
| `sse_frame_count` / `sse_body_bytes` | SSE JSON chunk 数与 body 字节数。 |
| `asgi_body_message_count` / `asgi_body_bytes` | `http.response.body` ASGI message 数与字节数。 |
| `disconnect_check_count` | HTTP disconnect 检查次数。 |
| `event_loop_lag` | 请求存活期间 loop lag monitor 的 p50/p95/p99/max。 |
| `engine_sampled_to_token_ready` | engine sampled ready -> token 对象 ready，包含增量 detokenize 和 `TokenOutput` 构建。 |
| `engine_decode` | 增量 tokenizer decode 调用本身。 |
| `engine_token_ready_to_loop_callback_scheduled` | engine 线程产生 token 到 callback 被调度。 |
| `loop_callback_wait` | callback scheduled -> callback started，近似 event loop ready callback 排队等待。 |
| `loop_callback_to_queue_put` | callback started -> 开始向 async queue 发布。 |
| `async_queue_put` | async queue put 调用耗时。 |
| `queue_put_to_queue_get` | queue item 驻留时间。 |
| `timer_cancel` | stream timeout timer cancel。 |
| `queue_task_done` | queue `task_done()`。 |
| `queue_get_to_route_received` | stream consumer 取到 item -> route 收到 token。 |
| `disconnect_check` | `http_request.is_disconnected()` 调用。 |
| `sse_json` | SSE chunk JSON 序列化。 |
| `route_yield_to_resumed` | generator yield 后挂起到恢复。 |
| `asgi_body_send` | ASGI response body send。 |

`token_trace` 一定保留首 token，之后按采样率随机保留，最多 32 条。trace 中的
timestamp 是原始 monotonic 纳秒；`durations_ms` 才是聚合计数和总量。

event loop lag monitor 只度量定时任务实际唤醒时间与目标时间的差值。它是
loop 忙碌或回调排队的间接信号，不是 ready callback 队列长度，也不能列出
队列中具体有哪些 callback。

`validate_logs.py` 和 `compute_metrics.py` 只处理 request timing JSONL，不读取
`/tmp/infinilm-output-timing`。输出 timing 需要单独解析 schema v1 JSONL。

## 相关性能开关

以下开关不是 request timing recorder 的配置，但会改变被测路径。做 A/B 或
和历史数据比较时必须显式固定它们。

| 环境变量 | 默认值 | 合法值 | 对 timing 的影响 |
|---|---|---|---|
| `INFINILM_PREPROCESS_WORKERS` | `1` | `1/2` | chat tokenize / token ids 转换使用的专用 worker 数。影响入口各分段和 event loop 占用。 |
| `INFINILM_OUTPUT_CALLBACK_COALESCE` | `true` | boolean | 合并多个 engine step 的输出 callback，减少 event loop callback 数。 |
| `INFINILM_OUTPUT_FLUSH_MAX_TOKENS` | `256` | `0/64/128/256/512` | 每次 coalesced flush 最多发布的 token 数；`0` 表示一次 flush 全部 pending。 |
| `INFINILM_OUTPUT_TOKEN_BATCH` | `true` | boolean | 同一 request 的多个 token 包装成一个 queue item。影响 `queue_put_to_queue_get` 的 item 数口径。 |
| `INFINILM_STREAM_READY_DRAIN_ITEMS` | `4` | `0/1/2/4/8` | stream consumer 醒来后额外 drain 多少个 ready queue item；`0` 表示每次只阻塞等待一个 item。 |
| `INFINILM_SSE_FLUSH_TOKENS` | `4` | `1/4/8` | 多少 token 合并一个 SSE chunk；首 token 和 finish/error 仍立即 flush。 |
| `INFINILM_DISCONNECT_CHECK_TOKENS` | `32` | `32/1024` | 每多少 token 检查一次 HTTP disconnect。影响输出路径 CPU 与 disconnect 响应时间。 |
| `INFINILM_MOONCAKE_DATA_TRANSFER_BYPASS` | `false` | boolean | 诊断开关：保留 bootstrap/ZMQ，跳过 Mooncake engine 数据写和 D ready flag 等待。开启后输出内容不保证正确。 |

除 `INFINILM_OUTPUT_TIMING` 的 truthy 解析外，上表 boolean 开关均要求显式
合法值；非法配置启动时失败。`INFINILM_MOONCAKE_DATA_TRANSFER_BYPASS=true`
必须同时设置在 P 和 D 进程上，只开一侧会破坏协议假设或导致 D 侧等待。

## 使用场景

### 1. 默认线上/常规压测

不额外配置时使用 sampled summary：

```bash
cd /workspace_codex/InfiniLM
bash run_server.sh
```

该模式保留入口分段、首 forward、decode forward 估算和 PD KV event，同时避免
逐 token/逐 step JSON 写放大。适合回答：

```text
入口处理、scheduler 等待、prefill/decode 计算和 KV readiness 哪项更大？
TTFT 与 D KV readiness 是否同步变化？
长时间运行的 timing 文件是否积压或丢弃？
```

如果只需要极限吞吐且不接受任何观测开销，可以额外设置：

```bash
export INFINILM_REQUEST_TIMING=false
```

但关闭后没有 request timing 证据，性能报告只能依赖客户端指标。

### 2. 精确 request 级 summary

用于 sampled estimate 与 off 之间的对照，或需要准确 per-request forward 总量：

```bash
export INFINILM_REQUEST_TIMING=true
export INFINILM_REQUEST_TIMING_MODE=summary
export INFINILM_REQUEST_TIMING_SAMPLE_INTERVAL=1
export INFINILM_REQUEST_TIMING_DIR=/tmp/infinilm-request-timing-exact
bash run_server.sh
```

该模式仍不写 step 记录，因此不能得到 batch histogram；但每个 decode forward
duration 都被累计，输出 schema v7。

### 3. 调度与物理 batch 诊断

需要以下任一问题时使用 full：

```text
decode batch 是否长期为 1
同一 forward_batch_id 中 batch size 是否异常
forward busy ratio 是否接近 100%
waiting/running/remote_kv backlog 如何变化
某一个 step 的 forward 或调度边界是否异常
```

启动：

```bash
export INFINILM_REQUEST_TIMING=true
export INFINILM_REQUEST_TIMING_MODE=full
export INFINILM_REQUEST_TIMING_DIR=/tmp/infinilm-request-timing-full
bash run_server.sh
```

full 会显著增加记录量和后台写压力，只建议短时间复现问题时使用。

### 4. 输出 event loop / SSE 路径诊断

当客户端 ITL、TPOT 或 stream callback 排队可疑时，与 request summary 同时开启：

```bash
export INFINILM_REQUEST_TIMING=true
export INFINILM_REQUEST_TIMING_MODE=summary
export INFINILM_REQUEST_TIMING_SAMPLE_INTERVAL=32

export INFINILM_OUTPUT_TIMING=true
export INFINILM_OUTPUT_TRACE_SAMPLE_RATE=0.01
export INFINILM_EVENT_LOOP_LAG_INTERVAL_MS=10
export INFINILM_OUTPUT_TIMING_DIR=/tmp/infinilm-output-timing

bash run_server.sh
```

分析顺序：

```text
1. request timing: engine forward 是否稳定
2. output timing: loop_callback_wait 与 queue_put_to_queue_get 是否增大
3. output timing: sse_json / asgi_body_send 是否增大
4. event_loop_lag 是否同步升高
5. 比较不同 OUTPUT_CALLBACK_COALESCE / OUTPUT_TOKEN_BATCH / SSE_FLUSH_TOKENS 配置
```

`process_summary.jsonl` 只有服务正常 shutdown 时写出；强制 kill 进程会丢失该
汇总。

### 5. PD 请求级分析

P/D 可以共用同一个 timing root：

```bash
export INFINILM_REQUEST_TIMING=true
export INFINILM_REQUEST_TIMING_MODE=summary
export INFINILM_REQUEST_TIMING_SAMPLE_INTERVAL=32
export INFINILM_REQUEST_TIMING_DIR=/tmp/infinilm-request-timing-pd

bash run_prefill.sh &
bash run_decode.sh &
# 等待 9103 / 9104 health
bash run_proxy.sh &
```

分析：

```bash
cd /workspace_codex/InfiniLM
PYTHONPATH=python python -m infinilm.llm.request_timing.validate_logs \
  --timing-root /tmp/infinilm-request-timing-pd \
  --format json > request_timing_validation.json

PYTHONPATH=python python -m infinilm.llm.request_timing.compute_metrics \
  --timing-root /tmp/infinilm-request-timing-pd \
  --format json > request_timing_metrics.json
```

如果 P/D 分别使用了不同 root，先构造只包含本次 P/D PID 目录的合并目录：

```bash
mkdir -p /tmp/infinilm-request-timing-merged
ln -s /tmp/prefill-timing/<prefill_pid> \
  /tmp/infinilm-request-timing-merged/<prefill_pid>
ln -s /tmp/decode-timing/<decode_pid> \
  /tmp/infinilm-request-timing-merged/<decode_pid>
```

warmup 和 smoke 请求也会进入 timing root。正式分析应按客户端结果或 finish 字段
过滤，例如 `prompt_tokens=1024`、`generated_tokens=1024`，不要直接把所有文件
都当成 formal requests。

### 6. Mooncake 数据面归因

在真实计算 PD 上做数据面 A/B：

```bash
export INFINILM_CLUSTER_COMMUNICATION_TEST=false
export INFINILM_REQUEST_TIMING=true
export INFINILM_REQUEST_TIMING_MODE=summary
export INFINILM_REQUEST_TIMING_SAMPLE_INTERVAL=32
export INFINILM_MOONCAKE_DATA_TRANSFER_BYPASS=true
```

分别运行 `true` 和 `false` 两组相同负载，然后比较：

```text
client TTFT / E2EL / throughput
P kv_send duration
D kv_receive duration
D decode forward estimate
decode_start_after_kv_ready_ms
```

`true` 组的生成内容不可用于正确性验证；`bytes` 和 `block_count` 只表示原本
应传输的逻辑规模。

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
summary finish 的 timestamp 单调性、采样计数和估算字段类型
```

校验器兼容早期没有 `role`、`scope`、`clock_domain` 字段的日志：缺失
`role` 时按事件或 phase 推断角色；但缺失关键时间戳或 duration 不一致仍
会判为 invalid。
新 schema 还会校验同一 `forward_batch_id` 的 step 数量、batch size 和
时间戳，以及同一 `kv_send batch_id` 的 request 数、总量和复制字段是否
一致。
同时会检查 `schema_version`、`forward_batch_id` 类型、文件名与 request id
的安全映射，以及 PD 中 D 首 decode forward 是否早于最终 KV receive 完成。
`forward_batch_id` 和 step 序列检查只适用于 full schema v6；summary schema
v7/v8 没有 step 记录，因此校验重点转为 finish 汇总自洽性。

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

### 计算函数

| 函数 | 含义 |
|---|---|
| `_role_metrics()` | 把一个 request 在 aggregated/prefill/decode 角色上的 full step 或 summary finish 转成统一指标。 |
| `_event_metrics()` | 提取请求级 `kv_send` / `kv_receive` 终态事件；同一事件多条时暂取最后一条。 |
| `_request_metrics()` | 关联 P/D 文件，输出 request 级 compute、入口路径和 KV readiness 指标。 |
| `_physical_forward_summary()` | 只在 full 模式有 step 时，按 `forward_batch_id` 去重并输出节点级物理 forward 统计。 |
| `_scheduler_snapshot_summary()` | full 模式下按物理 batch 去重 scheduler queue snapshot 和 active backlog。 |
| `_transfer_summary()` | 输出 request-weighted、batch-weighted、busy union 和 Mooncake 观测吞吐。 |
| `compute_timing_metrics()` | 先读取/校验 timing root，再输出 `summary` 与 `requests` 两层结果。 |

summary 模式的角色指标会额外带有：

```text
timing_mode=summary
sample_interval
engine_forward_estimated=true|false
sampled_decode_steps
batch_size_counts={}   # 无 step 记录，不能解释为 batch 恒为空
```

### 普通 aggregated 服务

```text
ttft_local_ms
  server receive -> 第一条 prefill engine forward 完成。

prefill_compute_ms
  第一条 prefill step 的 engine_forward。
  summary 模式来自 finish 中的精确 first prefill forward。

decode_compute_ms
  所有 decode step 的 engine_forward 总和。
  summary interval=1 时为精确总和；interval>1 时为估算总和。
  估算公式为 sampled decode average * decode_step_count。
```

### PD 服务

```text
p_compute_ms
  P 节点 prefill engine_forward。
  summary 模式仍为精确 first prefill forward。

d_compute_ms
  D 节点所有 decode engine_forward 总和。
  summary interval=1 时精确；interval>1 时估算。

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
  # 以上物理 forward 指标依赖 full step 记录

mooncake_transfer.kv_send
  unique_batch_count
  batch_weighted_duration_ms
  batch_bytes_total
  busy_union_ms
  throughput_during_busy_gbps
```

sampled summary 下节点级 `request_step_records=0`、`physical_forward_count=0`、
`batch_size_counts={}`，这是没有 step 日志的数据模式限制，不代表没有发生
GPU forward。需要 physical forward 统计时必须改用 `full`。

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

当 `mooncake_data_transfer_bypassed=true` 时，`compute_metrics` 仍会按公式
输出 `batch_bytes_total` 和 `throughput_during_busy_gbps`，但这些是逻辑规模
和非真实传输 interval 的比值，不能写入真实带宽结论。分析 bypass 数据时应
以 duration、TTFT、E2EL 和 throughput 对比为主。

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

full 模式下，recorder 为了减少文件 IO：

```text
首条 step 立即写出
后续 step 先保存在内存
达到 256 条未落盘 step 或距离上次写出 10 秒时强制 flush
finish 时写出剩余 step 和 finish 汇总
```

后台写队列最多保留 1024 个 pending job。达到上限时 recorder 丢弃新的观测
记录并递增 `dropped_records`，不会阻塞推理线程。因此 `dropped_records > 0`
表示磁盘写入速度暂时落后，不代表请求失败。

因此短请求通常只修改文件两次；长请求会按阈值分段写出。进程退出时，
recorder 会尽量 flush 未完成请求的缓冲 step。
summary 模式没有 step 缓冲；finish 和 KV event 直接提交到同一个后台写队列。

## Recorder 统计

运行时可通过 `RequestTimingFileRecorder.stats()` 读取：

```text
dropped_records   后台队列满时丢弃的观测记录数
pending_jobs      已提交且尚未完成的后台写任务数
buffered_requests 当前仍有未落盘 step 的 request 数
```

过期文件清理在 recorder 首次写入时立即执行一次，之后每小时执行一次；
默认保留 24 小时。
