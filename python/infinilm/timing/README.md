# InfiniLM Summary Timing

设置 `INFINILM_TIMING=true` 可以开启 timing。默认关闭。该值是本功能唯一通过
环境变量控制的配置，非法值会在启动时报错。

其余配置是模块级全局变量：

```python
from infinilm.timing import (
    TIMING_MODE,
    TIMING_SAMPLE_INTERVAL,
    TIMING_DIR,
)
```

默认值为：

```text
TIMING_MODE=summary
TIMING_SAMPLE_INTERVAL=32
TIMING_DIR=<项目根目录>/timing
```

当前支持 `cache_type="paged"` 的普通流式服务路径和 Mooncake PD 分离路径。
以下路径不支持 timing：

- `StaticScheduler`
- 非流式 `_chat`
- speculative decoding
- pipeline parallel

每个进程输出：

```text
<TIMING_DIR>/<pid>/timing.jsonl
```

JSONL 中包含五类事件：

- `batch`：一次物理 prefill / decode batch，包含 batch size；`request_ids`
  包含物理 batch 内全部请求。timing 开启时，这些请求都必须是流式请求。
- `token`：首 token、按采样间隔选中的 token、finished token。
- token 采样与 SSE 文本解耦：空文本 token 也会进入采样判断。
- `finish`：一个流式请求的汇总记录。
- `kv_send`：P 侧 Mooncake producer batch timing。
- `kv_receive`：D 侧 KV pull readiness timing。

当前 timing schema version 为 `3`。v2 将 loop 提交时间点改为调用
`call_soon_threadsafe()` 之前，并把字段重命名为
`loop_submit_started_at_ns`。
v3 增加 Mooncake timing event。

记录的 9 个时间戳为：

```text
service_entered_at_ns
scheduler_enqueue_before_at_ns
forward_started_at_ns
forward_completed_at_ns
loop_submit_started_at_ns
batch_put_started_at_ns
batch_put_completed_at_ns
route_token_received_at_ns
route_token_end_at_ns
```

Mooncake P 侧记录：

```text
kv_cache_send_ready_at_ns
batch_transfer_sync_write_started_at_ns
batch_transfer_sync_write_completed_at_ns
```

`kv_cache_send_ready_at_ns` 在 P 侧 request 的 KV blocks 准备好并置位
`send_meta.ready` 前记录。后两个时间点紧贴 Mooncake
`batch_transfer_sync_write()` 调用两侧，因此传输耗时不含 executor 排队。

Mooncake D 侧记录：

```text
kv_cache_pull_started_at_ns
kv_cache_pull_confirmed_at_ns
```

`kv_cache_pull_started_at_ns` 在 D 决定拉取远程 KV 时记录，timeout 重试时保留
第一次开始时间。`kv_cache_pull_confirmed_at_ns` 在 ZMQ response 全部处理完成且
调度器确认 KV 可用后记录。当前 MACA ready flags 被禁用，因此该确认点是 ZMQ
control plane 的成功确认，不是 CPU flag polling。

Mooncake 事件携带 `clock_domain="local-monotonic"` 和
`cross_node_comparable=false`。P/D 时间戳不能跨节点相减；离线工具只计算本侧
duration，或计算同一 D 进程内 `kv_cache_pull_confirmed_at_ns` 到首个 decode
forward 开始之间的时间。

## 低开销策略

timing 关闭时，普通 paged 服务不会创建 `ForwardTiming` / `TokenTiming`，也不会在
StepThread 或 API event loop 中读取 `monotonic_ns()`。

timing 开启时，token 观测按 `TIMING_SAMPLE_INTERVAL` 采样。只有首 token、
采样 token 和 finished token 会创建并记录 `TokenTiming`；未采样 token 不维护
输出路径时间戳。

后台写线程最多积累 128 条 JSONL record 后一次写入文件。写队列仍为有界队列，
满时丢弃观测数据，不阻塞推理路径。

数据校验和指标计算：

```bash
PYTHONPATH=python python scripts/timing/validate_timing.py --timing-root timing
PYTHONPATH=python python scripts/timing/compute_timing_metrics.py --timing-root timing
```

`INFINILM_TIMING=true` 时，`stream=false` 请求会在 HTTP 入口返回 400；
`_chat()` 和 engine scheduler 侧也会防御性报错，避免无 timing 的请求进入
timing-enabled batch。

分布型指标统一输出：

```text
count
avg_ms
min_ms
max_ms
p50_ms
p95_ms
p99_ms
```

除 token-level 和 batch-level 指标外，`compute_timing_metrics.py` 还输出
request-level 指标：

```text
request_first_token_e2e_ms
request_e2e_to_final_route_ms
request_prefill_forward_ms
request_enqueue_to_prefill_start_ms
```

`loop_submit_started_at_ns` 在调用 `call_soon_threadsafe()` 之前记录。该时间点
只表示 StepThread 开始向 API event loop 提交 callback，不表示 callback 已经执行。

`route_token_received_at_ns` 在 `stream_request()` 把 token 返回给 HTTP route
generator 时记录。`route_token_end_at_ns` 在 route 消费完该 token 后记录；
流式路径包含 SSE JSON 构造。
