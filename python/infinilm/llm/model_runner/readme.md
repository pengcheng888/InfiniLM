# ModelRunner 模块说明

`python/infinilm/llm/model_runner/` 负责 InfiniLM 的模型执行层。它包含三类实现：

1. 默认的单进程 `ModelRunner`；
2. 可选的独立进程 `ModelRunnerProcessClient` 与 worker；
3. 面向集群通信测试的轻量 `CommunicationModelRunner`。

所有实现都通过 `create_model_runner()` 创建，调用方不直接选择具体类。

## 1. 代码结构

```text
model_runner/
├── __init__.py
├── model_runner.py
├── model_runner_process.py
├── model_runner_channel.py
├── model_runner_wire.py
├── communication_model_runner.py
└── speculative_runner.py
```

### `__init__.py`

提供统一工厂：

```python
runner = create_model_runner(config, hf_config=hf_config)
```

选择顺序：

```text
INFINILM_MODEL_RUNNER_PROCESS=1
    -> ModelRunnerProcessClient

INFINILM_CLUSTER_COMMUNICATION_TEST=true
    -> CommunicationModelRunner

其他情况
    -> ModelRunner
```

进程模式优先级最高。worker 子进程中创建真实 runner 时会临时把
`INFINILM_MODEL_RUNNER_PROCESS` 置为 `0`，避免递归创建进程代理。

### `model_runner.py`

默认单进程执行器，主要职责：

- 初始化模型、GPU KV cache 和 processor；
- 构造模型输入；
- 调用 `model_engine.forward()`；
- 返回 `ModelRunnerOutput`；
- 记录 `ModelForwardTiming`；
- 集成 worker 侧 KV connector；
- 支持 speculative runner。

该实现没有跨进程数据面，`supports_non_block=False`。

### `model_runner_process.py`

进程模式的进程生命周期和 scheduler 侧代理，主要包含：

- `model_runner_process_enabled()`；
- `_worker_main()`；
- `ModelRunnerProcessClient`；
- `ModelRunnerFuture`；
- worker 启动握手；
- worker death 检测；
- response timeout；
- `release_requests()`；
- IPC timing 聚合与输出。

`ModelRunnerProcessClient` 对 scheduler 暴露与普通 runner 兼容的接口：

```python
runner.execute_model(
    scheduler_output,
    non_block=False,
)
```

同步模式返回 `ModelRunnerOutput`；`non_block=True` 返回
`ModelRunnerFuture`。

当前进程模式要求：

- `cache_type="paged"`；
- 纯文本请求；
- 可序列化的普通 Python wire 数据；
- 不跨进程传递 GPU tensor 或完整 `InferenceRequest` 对象。

### `model_runner_channel.py`

只负责进程间传输层，不依赖 `InferenceRequest`、`SchedulerOutput` 或
`ModelRunnerOutput`。包含：

- `PipeChannel`；
- `VLLMMessageQueueChannel`；
- `SharedMemoryRing`；
- IPC mode 配置；
- shared-memory slot 大小配置；
- busy-loop / spin 配置；
- `poll(0) + sched_yield()` 短窗口等待；
- ZMQ IPC endpoint 创建和清理。

`PipeChannel` 是当前默认和推荐传输方式。

`VLLMMessageQueueChannel` 是参考 vLLM `MessageQueue` 的实验实现：

- request/response 各一个 shared-memory ring；
- 每个 ring 2 个 slot；
- ZMQ IPC 只传 4 字节 slot id；
- payload 仍然 pickle；
- 支持 request/response 双向通知；
- 未实现 vLLM 的 remote reader、tensor arena、out-of-band tensor buffer
  和完整 overflow 机制。

当前实测该 shared-memory 实现相对 Pipe 没有端到端收益，仍保留为后续
zero-copy 协议的基础。

### `model_runner_wire.py`

只负责协议编解码，不负责 Pipe、ZMQ 或 shared-memory 传输。包含：

- request full snapshot；
- decode diff；
- worker 侧 `WireRequest`；
- batch wire layout；
- Mooncake connector metadata 编解码；
- `ModelRunnerOutput` response 编解码；
- connector-only 控制步轻量 response 编解码；
- request cache 和 fixed array 配置。

进程模式的数据边界：

```text
Scheduler 侧
    InferenceRequest / SchedulerOutput
        |
        | batch_to_wire()
        v
普通 Python dict / list / array
        |
        | model_runner_channel
        v
Worker 侧
    WireRequest / SchedulerOutput
        |
        | 真实 ModelRunner
        v
ModelRunnerOutput
        |
        | response_to_wire()
        v
Scheduler 侧
```

#### request cache

默认开启。请求第一次调度时发送 full snapshot：

```text
prompt_token_ids
完整 generated_token_ids
block_table
slot_mapping
采样参数
KV transfer 参数
```

后续 decode 只发送增量：

```text
request_id
新增 generated token
新的 block/slot 状态
少量数值状态
```

worker 复用 `WireRequest` 对象。请求结束后，scheduler 调用
`release_requests()` 清理父子两侧缓存。如果生成 token 长度回退，协议会
自动降级为 full snapshot。

#### connector-only 控制步

P/D 场景中，scheduler 可能在没有 scheduled request 时仍需要查询 Mooncake
发送/接收完成事件。这类 step 不构造 ModelRunner batch，而是走专用
`connector_control` 命令：

```text
metadata_to_wire()       # 空 metadata 时为 None
    -> connector_control
    -> worker execute_connector_control()
    -> connector_control_response_to_wire()
```

控制步没有 request cache diff、sampled token 和 forward timing。如果
connector output 为空，response 使用 `None` 表示，避免完整空
`ModelRunnerOutput` 序列化。

#### fixed array layout

`INFINILM_MODEL_RUNNER_FIXED_ARRAY=1` 会把 decode diff 编码为平铺
`array('q')` 和 offsets。

该布局为 shared-memory / zero-copy 协议预留。当前 Pipe + pickle 路径下，
每次创建 array 和 offsets 会抵消 payload 缩减收益，实测为负优化，因此
默认关闭。

### `communication_model_runner.py`

`INFINILM_CLUSTER_COMMUNICATION_TEST=true` 时选择的轻量 runner，用于验证
集群通信，不加载真实大模型。它支持假 forward 和 KV connector 收发状态
收集，`supports_non_block=False`。

### `speculative_runner.py`

 speculative decoding 的内部执行器，由普通 `ModelRunner` 使用。它不是
scheduler 直接创建的 runner 类型。

## 2. 进程模式执行流程

### 初始化

```text
LLMEngine
    -> create_model_runner()
    -> ModelRunnerProcessClient
    -> spawn ModelRunner worker
    -> worker 创建真实 ModelRunner
    -> worker 返回 effective_num_blocks / EOS
    -> scheduler 使用 worker 同步回来的容量初始化
```

启动握手仍使用控制面 `multiprocessing.Pipe`。数据面由
`INFINILM_MODEL_RUNNER_IPC` 决定。

### 执行

```text
LLMEngine.step()
    -> scheduler.schedule()

connector-only 分支:
    -> channel.send()
    -> metadata_from_wire()
    -> 真实 ModelRunner.execute_connector_control()
    -> connector_control_response_to_wire()
    -> channel response
    -> ModelRunnerFuture.result()
    -> scheduler.update_from_output()

prefill/decode 分支:
    -> batch_to_wire()
    -> channel.send()
    -> worker recv
    -> requests_from_wire()
    -> 真实 ModelRunner.execute_model()
    -> response_to_wire()
    -> channel response
    -> ModelRunnerFuture.result()
    -> response_from_wire()
    -> scheduler.update_from_output()
```

`LLMEngine` 通过 runner 的 `supports_connector_control` 显式判断是否可用
轻量控制路径；单进程真实 runner 和进程代理声明为 true，轻量通信测试 runner
声明为 false。

### prefill / decode 调度策略

当 `INFINILM_MODEL_RUNNER_BATCH_QUEUE_SIZE > 1` 时：

- prefill batch 同步执行；
- connector-only 控制步骤同步执行；
- decode batch 可以 `non_block=True` 进入 in-flight 队列；
- 已有 in-flight decode 且来了新 prefill 时，先 commit decode，再调度
  prefill；
- decode response 严格按提交顺序 commit。

该策略避免早期 queue=2 将请求固化成 `1 + 7` 两个小 decode group。

`INFINILM_MODEL_RUNNER_BATCH_QUEUE_SIZE=1` 是安全默认值，表示进程模式不
使用第二个 in-flight decode batch。

## 3. 环境变量

### 进程模式

| 环境变量 | 默认值 | 可选值 | 说明 |
|---|---:|---|---|
| `INFINILM_MODEL_RUNNER_PROCESS` | `0` | `0/1/false/true/no/yes/off/on` | 是否启用独立 ModelRunner 进程 |

注意：代码默认值是 `0`。`run_server.sh`、`run_prefill.sh` 和
`run_decode.sh` 是 `/workspace_codex/InfiniLM` 下的本地部署脚本，当前被
`.gitignore` 的 `*.sh` 规则排除；这些脚本会默认设置为 `1`，但这不是
Python 包自身的默认配置。

### decode batch queue

| 环境变量 | 默认值 | 可选值 | 说明 |
|---|---:|---:|---|
| `INFINILM_MODEL_RUNNER_BATCH_QUEUE_SIZE` | `1` | `1~4` | 进程模式 decode in-flight 深度 |

该队列只保存 decode batch。深度大于 1 时 prefill 仍同步执行。

推荐：

```text
安全默认：1
普通服务高吞吐实验：2
```

不建议直接使用 3 或 4；更高深度需要重新验证 batch 形成和输出延迟。

### IPC 传输

| 环境变量 | 默认值 | 可选值 | 说明 |
|---|---:|---|---|
| `INFINILM_MODEL_RUNNER_IPC` | `pipe` | `pipe/shared_memory/vllm_message_queue` | 数据面传输模式 |

`shared_memory` 是 `vllm_message_queue` 的别名。

当前推荐：

```text
pipe
```

### shared-memory slot

| 环境变量 | 默认值 | 约束 | 说明 |
|---|---:|---:|---|
| `INFINILM_MODEL_RUNNER_SHM_SLOT_BYTES` | `8388608` | `>=65536` | 每个 shared-memory slot 的字节数 |

仅在 `INFINILM_MODEL_RUNNER_IPC=vllm_message_queue` 时生效。

当前实现包含 request 和 response 两个 ring，每个 ring 2 个 slot。默认
8 MiB slot 时，总 payload 空间约为：

```text
2 request slots * 8 MiB + 2 response slots * 8 MiB = 32 MiB
```

### spin / notify

| 环境变量 | 默认值 | 可选值 | 说明 |
|---|---:|---:|---|
| `INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US` | `0` | `0~1000000` | 短窗口 busy-poll 微秒数 |

行为：

```text
等待开始
    -> poll(0) + sched_yield() 忙等
    -> 超过窗口仍无数据
    -> 回落阻塞 poll/recv
```

该配置同时作用于 Pipe 和 shared-memory channel。

本地部署脚本 `run_server.sh` 当前默认设置为 `20000`；该脚本被当前
`.gitignore` 排除，不随 Python 包代码发布。P/D 本地脚本没有默认开启，
P/D 场景应先使用 `0` 验证，再单独评估 CPU 干扰。

### request cache

| 环境变量 | 默认值 | 可选值 | 说明 |
|---|---:|---|---|
| `INFINILM_MODEL_RUNNER_REQUEST_CACHE` | `1` | `0/1/false/true/no/yes/off/on` | worker 缓存请求快照，decode 只传增量 |

该配置只在进程模式下生效。普通单进程 runner 没有跨进程重复传输问题。

### fixed array wire layout

| 环境变量 | 默认值 | 可选值 | 说明 |
|---|---:|---|---|
| `INFINILM_MODEL_RUNNER_FIXED_ARRAY` | `0` | `0/1/false/true/no/yes/off/on` | 是否使用平铺 int array decode layout |

实验配置。当前 Pipe + pickle 路径下不建议开启。

### 集群通信测试模式

| 环境变量 | 默认值 | 可选值 | 说明 |
|---|---:|---|---|
| `INFINILM_CLUSTER_COMMUNICATION_TEST` | `false` | boolean | 使用轻量通信测试 runner |

该模式只在未启用进程模式时生效。

### 调试 timing

这些变量不属于 ModelRunner 数据面，但常用于分析进程模式性能：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `INFINILM_REQUEST_TIMING` | `true` | 是否记录 request timing |
| `INFINILM_REQUEST_TIMING_MODE` | `summary` | `summary/full` |
| `INFINILM_REQUEST_TIMING_SAMPLE_INTERVAL` | `32` | `1/4/8/16/32/64/128` |
| `INFINILM_REQUEST_TIMING_DIR` | `/tmp/infinilm-request-timing` | timing 输出目录 |

`full + interval=1` 会产生大量 JSONL 数据，适合定位问题，不适合长期生产
运行。

## 4. 常用配置

### 4.1 单进程基线

最保守配置：

```bash
INFINILM_MODEL_RUNNER_PROCESS=0
INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US=0
```

特点：

- 无跨进程 IPC；
- 行为最简单；
- scheduler 与 ModelRunner 共享同一 Python 进程；
- 适合作为性能和正确性基线。

### 4.2 安全进程模式

隔离 scheduler 和 ModelRunner，但不使用第二个 decode in-flight batch：

```bash
INFINILM_MODEL_RUNNER_PROCESS=1
INFINILM_MODEL_RUNNER_BATCH_QUEUE_SIZE=1
INFINILM_MODEL_RUNNER_REQUEST_CACHE=1
INFINILM_MODEL_RUNNER_IPC=pipe
INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US=0
INFINILM_MODEL_RUNNER_FIXED_ARRAY=0
```

特点：

- 避免共享 GIL；
- request cache 降低跨进程 payload；
- 无额外 busy-poll CPU 占用；
- 调度行为接近 queue=1。

### 4.3 普通服务高性能配置

在 2026-11-03 的 Qwen3-8B、100 请求、并发 8、输入输出 1024 tokens 测试中，
以下配置吞吐和 ITL 最好：

```bash
INFINILM_MODEL_RUNNER_PROCESS=1
INFINILM_MODEL_RUNNER_BATCH_QUEUE_SIZE=2
INFINILM_MODEL_RUNNER_REQUEST_CACHE=1
INFINILM_MODEL_RUNNER_IPC=pipe
INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US=20000
INFINILM_MODEL_RUNNER_FIXED_ARRAY=0
```

启动示例：

```bash
cd /workspace_codex/InfiniLM

env \
  INFINILM_MODEL_RUNNER_PROCESS=1 \
  INFINILM_MODEL_RUNNER_BATCH_QUEUE_SIZE=2 \
  INFINILM_MODEL_RUNNER_REQUEST_CACHE=1 \
  INFINILM_MODEL_RUNNER_IPC=pipe \
  INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US=20000 \
  INFINILM_MODEL_RUNNER_FIXED_ARRAY=0 \
  INFINILM_REQUEST_TIMING=false \
  bash run_server.sh
```

注意：

- 本地部署脚本 `run_server.sh` 已默认开启进程模式和 20ms spin；
- queue size 代码默认是 1，高吞吐配置必须显式设置 `2`；
- 20ms spin 会提高 CPU 占用；
- 该结论当前只针对普通服务，不自动适用于 P/D。

### 4.4 P/D 服务建议初始配置

建议先使用保守配置：

```bash
INFINILM_MODEL_RUNNER_PROCESS=1
INFINILM_MODEL_RUNNER_BATCH_QUEUE_SIZE=1
INFINILM_MODEL_RUNNER_REQUEST_CACHE=1
INFINILM_MODEL_RUNNER_IPC=pipe
INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US=0
INFINILM_MODEL_RUNNER_FIXED_ARRAY=0
```

原因：

- P/D 侧存在 Mooncake TCP/RDMA 传输；
- scheduler 和 connector 本身有 CPU 负载；
- busy-poll 可能干扰通信线程或 event loop；
- KV readiness 事件比普通服务更敏感。

如需测试 P/D queue=2 或 spin，应单独压测并比较：

```text
TTFT
Mean ITL / TPOT
D KV readiness
P TransferEngine 调用耗时
Mooncake timing event
CPU 使用率
失败请求数
```

### 4.5 shared-memory 实验配置

```bash
INFINILM_MODEL_RUNNER_PROCESS=1
INFINILM_MODEL_RUNNER_BATCH_QUEUE_SIZE=2
INFINILM_MODEL_RUNNER_REQUEST_CACHE=1
INFINILM_MODEL_RUNNER_IPC=vllm_message_queue
INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US=20000
INFINILM_MODEL_RUNNER_FIXED_ARRAY=0
```

该配置用于验证 shared-memory ring 和通知机制。当前普通服务实测与 Pipe
基本持平，不是默认推荐配置。

## 5. 性能和调试建议

### 5.1 判断配置是否生效

服务日志会输出进程模式相关行为。完整压测结束关闭服务时，进程模式会输出：

```text
ModelRunner IPC timing summary: {...}
```

其中包含：

```text
batches
execute_batches
connector_control_batches
connector_metadata_empty
connector_metadata_nonempty
connector_empty_output
connector_nonempty_output
connector_finished_sending_requests
connector_finished_recving_requests
connector_failed_recving_requests
requests
full_requests
diff_requests
parent_wire
parent_send
send_start_to_worker_recv
request_rebuild
metadata_from_wire
worker_recv_to_execute_start
runner_execute
response_wire
parent_response_decode
response_send_start_to_parent_recv
submit_to_response_decoded
```

connector-only 控制步的耗时统计使用 `connector_control_` 前缀，例如
`connector_control_runner_execute`；普通 execute batch 保持原有字段名。

这些统计用于定位跨进程通信和 forward 准备开销。

### 5.2 常见问题定位

#### decode batch 被拆小

优先检查：

```text
INFINILM_MODEL_RUNNER_BATCH_QUEUE_SIZE
decode batch size 分布
producer_batches
```

queue=2 必须配合 prefill 同步策略。当前实现已经保证 prefill 不进入
decode in-flight 队列。

#### 进程模式明显慢于单进程

优先确认：

```bash
INFINILM_MODEL_RUNNER_REQUEST_CACHE=1
INFINILM_MODEL_RUNNER_IPC=pipe
INFINILM_MODEL_RUNNER_FIXED_ARRAY=0
```

再查看 IPC summary 中：

```text
parent_wire
send_start_to_worker_recv
request_rebuild
response_send_start_to_parent_recv
```

#### CPU 敏感场景变慢

检查：

```bash
INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US
```

先设置为 `0`。普通服务可尝试 `1000~50000` 之间扫描，但需要同时观察吞吐
和 CPU 占用。

#### shared-memory 模式消息过大

会出现类似错误：

```text
ModelRunner IPC message is ... bytes
```

处理方式：

```bash
INFINILM_MODEL_RUNNER_SHM_SLOT_BYTES=<larger-value>
```

或临时切回：

```bash
INFINILM_MODEL_RUNNER_IPC=pipe
```

### 5.3 测试建议

修改本目录代码后，建议至少执行：

```bash
cd /workspace_codex/InfiniLM
pytest -q test/service/test_model_runner_process.py
python -m py_compile \
  python/infinilm/llm/model_runner/model_runner.py \
  python/infinilm/llm/model_runner/model_runner_channel.py \
  python/infinilm/llm/model_runner/model_runner_wire.py \
  python/infinilm/llm/model_runner/model_runner_process.py
```

涉及进程启动、通信或 P/D connector 时，还需要分别 smoke 测试：

```text
process=false 普通服务
process=true queue=1 普通服务
process=true queue=2 普通服务
pipe 模式
vllm_message_queue 模式
P/D proxy/prefill/decode
```

## 6. 扩展约定

新增通信机制时，遵循以下边界：

```text
Pipe / ZMQ / shared-memory / spin
    -> model_runner_channel.py

request / batch / response 编解码
    -> model_runner_wire.py

进程启动、关闭、Future、timeout、worker death
    -> model_runner_process.py

真实 forward
    -> model_runner.py
```

不要在 channel 层 import 推理领域对象，也不要在 wire 层直接操作 socket 或
shared memory。这样后续替换为 vLLM 风格 zero-copy MessageQueue 时，可以
只修改 channel 和 wire layout，不影响进程生命周期逻辑。
