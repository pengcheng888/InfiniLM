# ModelRunner Connector Completion Event 优化方案

## 1. 背景

当前 `process=true` 的 PD 服务中，scheduler 每约 3ms 调用一次
`Scheduler.schedule()`。当 Mooncake connector 存在、但当前没有新的 prefill/decode
请求时，scheduler 仍可能返回一个空的 `SchedulerOutput`。

随后会进入以下路径：

```text
Scheduler.schedule()
    -> 空 SchedulerOutput
    -> LLMEngine._execute_synchronous_step()
    -> ModelRunnerProcessClient.execute_connector_control()
    -> worker.get_finished()
    -> Pipe request/response
```

当前 connector control fast path 已经减少了单次 IPC 的 payload 和编解码成本，但没有
减少空 connector control 的调用次数。大量空调用仍然会产生：

- parent -> worker 的 IPC；
- worker 侧 metadata 解码和 `get_finished()` 调用；
- worker -> parent 的 response；
- parent response 解码；
- scheduler 和 ModelRunner 之间的调度噪声。

本方案保留现有每 3ms 一次的 scheduler 调度和 CPU 空转，但不再每个空 step 都发起
connector IPC。

## 2. 目标行为

connector-only IPC 只在以下三种情况发生：

```text
1. 当前 step 携带新的 connector metadata，需要启动 KV transfer；
2. 收到了 Mooncake completion notification，需要立即 drain 完成事件；
3. 距离上次 connector completion 检查已经超过 1000ms，执行一次兜底检查。
```

其余情况直接返回，不调用 `execute_connector_control()`：

```text
每 3ms 仍可调用 scheduler
        |
        +-- 有 metadata       -> connector IPC
        +-- 有 completion event -> connector IPC
        +-- 超过 1000ms       -> connector IPC
        +-- 其他情况          -> 跳过 connector IPC
```

1000ms 只用于防止通知丢失或通知通道异常，正常路径应由 completion notification
驱动。通知正常时，KV completion 不需要等待 1000ms。

## 3. 进程边界限制

worker 进程不能直接修改父进程 `ModelRunnerProcessClient` 中的变量：

```python
self.a = True
```

因为两个进程拥有不同的地址空间。worker 必须通过独立的 completion notification
channel 发送事件，父进程收到事件后再更新本地状态。

建议通信结构如下：

```text
request channel       parent -> worker
response channel      worker -> parent
completion channel    worker -> parent
```

completion channel 应独立于现有 response channel。Mooncake sender/receiver 是后台
线程，不能直接与 worker 主循环并发写同一个 multiprocessing `Connection`。建议由
worker 内部的 notifier thread 串行发送 completion notification。

## 4. 父进程状态设计

不建议只维护一个未经保护的布尔变量：

```python
self.a = False
```

推荐使用 `threading.Event` 加 generation counter：

```python
self._connector_completion_event = threading.Event()
self._connector_completion_generation = 0
self._connector_consumed_generation = 0
self._connector_completion_lock = threading.Lock()

self._connector_poll_interval_s = 1.0
self._connector_last_poll_at = time.monotonic()
```

其中：

- `Event` 用于快速判断是否有 completion notification；
- `generation` 用于识别多个合并到达的通知；
- `Lock` 保护 generation 和消费状态；
- `last_poll_at` 用于 1000ms 定时兜底；
- `poll_interval_s=1.0` 表示 1000ms。

收到通知时，父进程 reader thread 执行：

```python
with self._connector_completion_lock:
    self._connector_completion_generation += 1
    self._connector_completion_event.set()
```

## 5. connector IPC 判断逻辑

建议在 `ModelRunnerProcessClient` 中提供统一判断接口：

```python
def should_poll_connector(self, metadata) -> bool:
    if metadata is not None:
        return True

    with self._connector_completion_lock:
        if (
            self._connector_completion_generation
            != self._connector_consumed_generation
        ):
            return True

        return (
            time.monotonic() - self._connector_last_poll_at
            >= self._connector_poll_interval_s
        )
```

判断优先级为：

```text
metadata 非空
    -> 立即执行

有未消费 completion generation
    -> 立即执行

距离上次检查 >= 1000ms
    -> 执行一次兜底检查

其他情况
    -> 跳过 IPC
```

metadata 非空时不能跳过，因为该调用可能负责：

- 启动 KV receive；
- 启动 KV send；
- 注册远端 block 信息；
- 处理新的 transfer request。

## 6. 避免清理事件时丢事件

不能简单地在 control 完成后执行：

```python
self._connector_completion_event.clear()
```

因为可能发生以下竞态：

```text
step 开始 connector control
completion reader 收到新事件
connector control 返回
step 清除 Event
```

这样会把 control 期间到达的新事件清掉。

推荐在 connector control 开始前保存 generation：

```python
with self._connector_completion_lock:
    observed_generation = self._connector_completion_generation
    self._connector_last_poll_at = time.monotonic()
```

control 完成后只消费已经观察到的 generation：

```python
with self._connector_completion_lock:
    self._connector_consumed_generation = max(
        self._connector_consumed_generation,
        observed_generation,
    )

    if (
        self._connector_completion_generation
        == self._connector_consumed_generation
    ):
        self._connector_completion_event.clear()
```

如果 control 期间有新通知，generation 会更大，Event 保持设置状态，下一次 step
会继续执行 connector control。

## 7. ModelRunnerProcessClient 修改位置

主要修改文件：

```text
python/infinilm/llm/model_runner/model_runner_process.py
```

建议增加以下接口：

```python
def has_connector_completion(self) -> bool:
    with self._connector_completion_lock:
        return (
            self._connector_completion_generation
            != self._connector_consumed_generation
        )

def should_poll_connector(self, metadata) -> bool:
    ...

def mark_connector_polled(self, observed_generation: int) -> None:
    ...
```

completion reader thread 负责接收独立通道的通知，不负责执行 scheduler 更新。这样
可以保持 scheduler 状态只在 engine step 线程中修改。

## 8. LLMEngine 接入位置

主要修改文件：

```text
python/infinilm/llm/llm.py
```

`_execute_synchronous_step()` 中的 connector-only 分支应当先判断是否需要 IPC：

```python
if not is_prefill and not scheduler_output.scheduled_requests:
    metadata = scheduler_output.kv_connector_metadata
    if not self.model_runner.should_poll_connector(metadata):
        return ModelRunnerOutput()

    return self.model_runner.execute_connector_control(scheduler_output)
```

实际实现中建议将 generation 记录和 `last_poll_at` 更新时间封装在
`execute_connector_control()` 内部，避免不同调用路径遗漏状态更新。

## 9. `step_with_batch_queue()` 的特殊处理

不能在 `step_with_batch_queue()` 开头无条件因为 connector event 未设置就返回，
因为队列中可能仍然存在已经提交的 decode Future。

正确顺序应保持为：

```text
1. 处理已经完成的 decode Future；
2. 按 FIFO 规则提交 in-flight decode batch；
3. 调用 scheduler.schedule()；
4. prefill/decode batch 正常执行；
5. connector-only step 按 metadata/event/1000ms 判断是否执行 IPC；
6. 没有必要的 connector IPC 时返回 False。
```

1000ms 判断只适用于：

```python
not scheduler_output.scheduled_requests
and scheduler_output.kv_connector_metadata is None
```

不能影响普通模型 forward，也不能阻止带 metadata 的 KV 操作。

## 10. 普通 execute response 的时间更新

普通 prefill/decode 的 `execute_model()` 在 worker 内部也会调用 connector output
收集逻辑。如果该调用已经执行了 `get_finished()`，父进程应把它视为一次完成检查，
更新：

```python
self._connector_last_poll_at = time.monotonic()
```

否则可能出现：

```text
普通 decode 每隔很短时间执行一次
父进程却认为已经超过 1000ms 没有检查
额外产生一次 connector_control
```

建议在普通 execute response 和 connector control response 解码完成后，统一调用
一个 `_note_connector_observed()` 方法。

## 11. 与当前 3ms 调度空转的关系

本方案暂时不修改：

```python
time.sleep(0.003)
```

因此 scheduler 仍然约每 3ms 被调用一次，CPU 空转成本仍然存在。但 connector IPC
不再与 scheduler 轮询频率绑定：

```text
scheduler 调度频率：约 3ms
connector 空轮询频率：最多每 1000ms
connector 正常触发：completion notification 到达时
```

这是一个过渡方案，先解决高频 IPC 和 worker `get_finished()` 调用，再单独评估是否
需要把 scheduler 空闲等待也改为事件唤醒。

## 12. 预期收益和代价

预期收益：

- connector-only IPC 次数显著下降；
- 空 metadata 的 Pipe request/response 大幅减少；
- worker 侧 `get_finished()` 调用次数下降；
- scheduler、ModelRunner 和 connector 之间的调度噪声下降；
- process=true 的 PD 服务 CPU 负载降低。

代价和边界：

- completion notification channel 需要额外的进程间通道；
- 通知异常时，完成事件最坏延迟约 1000ms；
- 3ms scheduler 空转仍然保留；
- 需要处理 worker 退出、Pipe 关闭和通知 reader 错误；
- 必须通过 generation counter 避免并发清理造成事件丢失。

正常通知路径下，1000ms 不会增加 KV readiness 延迟；只有通知丢失或通知通道异常时，
才会退化为 1000ms 兜底检查。

## 13. 建议统计项

为验证优化效果，建议新增以下统计：

```text
connector_completion_notifications
connector_event_triggered_polls
connector_timer_triggered_polls
connector_skipped_empty_polls
connector_control_batches
connector_empty_control_batches
connector_completion_event_lag
connector_completion_coalesced
```

重点比较：

```text
connector_control_batches / 请求数
completion notifications / 请求数
timer fallback 次数
空 control 占比
KV completion notification 到 scheduler commit 的延迟
TTFT、TPOT、E2EL 和 output throughput
```

## 14. 推荐实施顺序

1. 增加独立 completion notification channel；
2. 父进程使用 `Event + generation counter` 接收通知；
3. 增加 1000ms fallback 判断；
4. 无 metadata、无通知且未超时的空 step 跳过 connector IPC；
5. 普通 execute response 和 connector control response 统一更新时间戳；
6. 增加 IPC 次数和 fallback 次数统计；
7. 测试通知正常、通知延迟、通知丢失、worker 退出和 shutdown 场景；
8. 根据测试结果再决定是否进一步移除 1000ms fallback 或改成直接推送完成 payload。

当前阶段建议只做 completion notification + connector control drain，不立即把完整
`KVConnectorOutput` 从 worker 直接推送到父进程。这样可以复用现有
`get_finished()` 消费逻辑，降低完成事件重复消费、乱序和 timing 数据丢失的风险。

## 15. 2026-11-04 实测补充

该方案已实现并测试。结果：

```text
connector-only IPC 次数大幅下降
TCP PD 端到端性能未提升
```

| 配置 | duration | output throughput | mean TTFT |
|---|---:|---:|---:|
| fast path 基线 | 252.63 s | 405.34 tok/s | 1,694.20 ms |
| event + 1000ms fallback | 255.88 s | 400.18 tok/s | 2,417.65 ms |
| event + 50ms fallback | 260.16 s | 393.61 tok/s | 2,888.95 ms |

因此当前实现默认关闭：

```bash
INFINILM_MODEL_RUNNER_CONNECTOR_COMPLETION_EVENT=0
```

如需实验开启：

```bash
INFINILM_MODEL_RUNNER_CONNECTOR_COMPLETION_EVENT=1
INFINILM_MODEL_RUNNER_CONNECTOR_POLL_INTERVAL_MS=1000
```

详细测试见：

[benchmark_infinilm_pd_tcp_connector_completion_event_20261104.md](/workspace_codex/InfiniLM-Test/benchmark_infinilm_pd_tcp_connector_completion_event_20261104.md)

## 16. event=0/1 交替复测结论

为确认 event 策略是否稳定造成性能衰减，又在当前最新代码上重新执行了一组
严格 A/B。两组除以下开关外配置完全一致：

```text
INFINILM_MODEL_RUNNER_CONNECTOR_COMPLETION_EVENT=0/1
INFINILM_MODEL_RUNNER_CONNECTOR_POLL_INTERVAL_MS=1000
INFINILM_MODEL_RUNNER_PROCESS=true
INFINILM_MODEL_RUNNER_BATCH_QUEUE_SIZE=2
INFINILM_MODEL_RUNNER_REQUEST_CACHE=1
INFINILM_MODEL_RUNNER_IPC=pipe
INFINILM_MODEL_RUNNER_FIXED_ARRAY=0
INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US=20000
MOONCAKE_PROTOCOL=tcp
```

请求矩阵：

```text
100 requests
concurrency=8
input/output=1024
request timing=full
```

结果目录：

- [event=0 client](/workspace_codex/InfiniLM-Test/benchmark_infinilm_pd_tcp_completion_event_recheck0_20261104)
- [event=1 client](/workspace_codex/InfiniLM-Test/benchmark_infinilm_pd_tcp_completion_event_recheck1_20261104)
- [event=0 P timing](/workspace_codex/InfiniLM-Test/request_timing_pd_tcp_completion_event_recheck0_prefill_20261104)
- [event=0 D timing](/workspace_codex/InfiniLM-Test/request_timing_pd_tcp_completion_event_recheck0_decode_20261104)
- [event=1 P timing](/workspace_codex/InfiniLM-Test/request_timing_pd_tcp_completion_event_recheck1_prefill_20261104)
- [event=1 D timing](/workspace_codex/InfiniLM-Test/request_timing_pd_tcp_completion_event_recheck1_decode_20261104)

### 16.1 Client 复测结果

两组均为 `100/100` 成功，全部生成 `1024` tokens。

| 指标 | event=0 | event=1 | event=1 - event=0 |
|---|---:|---:|---:|
| Benchmark duration | 256.06 s | 256.56 s | +0.50 s，+0.20% |
| Mean TTFT | 1,919.09 ms | 2,694.07 ms | +774.98 ms，+40.38% |
| Median TTFT | 1,931.72 ms | 2,340.39 ms | +408.68 ms |
| P90 TTFT | 2,010.60 ms | 4,171.47 ms | +2,160.87 ms |
| P99 TTFT | 2,120.57 ms | 6,780.21 ms | +4,659.64 ms |
| Mean TPOT | 17.41 ms | 16.93 ms | -0.48 ms |
| Mean ITL | 69.55 ms | 67.65 ms | -1.90 ms |
| Mean E2EL | 19,727.89 ms | 20,015.74 ms | +287.84 ms，+1.46% |
| P99 E2EL | 19,952.54 ms | 24,016.04 ms | +4,063.50 ms |
| Output throughput | 399.91 tok/s | 399.12 tok/s | -0.79 tok/s，-0.20% |

本轮结论需要区分两个层次：

1. 整体吞吐差异只有 `0.20%`，单轮看不能证明稳定吞吐回归；
2. TTFT、P90/P99 TTFT 和 P99 E2EL 的衰减复现非常明显，说明 event=1
   确实改变了 P/D readiness 和调度节奏。

结合前一轮结果：

| 轮次 | event=0 throughput | event=1 throughput | event=1 降幅 |
|---|---:|---:|---:|
| 前一轮 | 405.34 tok/s | 400.18 tok/s | -1.27% |
| 本轮复测 | 399.91 tok/s | 399.12 tok/s | -0.20% |
| 两轮平均 | 402.63 tok/s | 399.65 tok/s | -0.74% |

因此更稳妥的结论是：

```text
event=1 存在明确的 TTFT/readiness/batch 调度衰减；
整体吞吐有轻微负向趋势，但单轮吞吐差异需要重复测试确认。
```

### 16.2 KV readiness 与 P/D 对齐

| 指标 | event=0 | event=1 |
|---|---:|---:|
| D KV readiness mean | 1,856.86 ms | 2,572.77 ms |
| D KV readiness median | 1,881.43 ms | 2,258.75 ms |
| D KV readiness P90 | 1,950.65 ms | 4,091.85 ms |
| D KV readiness P99 | 1,965.26 ms | 6,668.12 ms |
| P send end -> D ready end mean | 35.05 ms | 1,727.30 ms |
| P send end -> D ready end median | 13.49 ms | 1,779.15 ms |
| P send end -> D ready end P90 | 15.03 ms | 3,577.21 ms |
| P send end -> D ready end P99 | 265.89 ms | 5,857.79 ms |

D KV readiness mean 增加：

```text
2,572.77 - 1,856.86 = 715.91 ms
```

涨幅约 `38.56%`。

P send 完成到 D ready 完成的 mean 从 `35.05 ms` 增至 `1,727.30 ms`，
说明主要问题仍发生在 KV readiness 暴露和 Mooncake receiver 调度路径，
而不是 client 输出或 decode forward 本身。

### 16.3 Batch 分布

P 侧：

| 指标 | event=0 | event=1 |
|---|---:|---:|
| Prefill batch 数 | 28 | 75 |
| batch size=1 | 14 | 62 |
| Prefill forward 总和 | 12,269.95 ms | 13,245.02 ms |

D 侧：

| 指标 | event=0 | event=1 |
|---|---:|---:|
| Decode batch 数 | 13,416 | 14,987 |
| batch size=8 | 12,262 | 8,725 |
| Decode forward 总和 | 210,852.54 ms | 233,997.95 ms |

D decode forward 总和增加：

```text
233,997.95 - 210,852.54 = 23,145.41 ms
```

增加约 `10.98%`。这复现了 event=1 的主要问题：P 侧 prefill 和 D 侧
decode batch 都明显变碎。

### 16.4 IPC 次数与 fallback

event=0：

```text
P execute batches:             28
P connector control batches: 102,841
D execute batches:          13,416
D connector control batches: 52,290
```

event=1：

```text
P execute batches:             75
P connector control batches:   443
P skipped empty polls:      117,508
P completion notifications:    108
P event-triggered polls:       62
P timer-triggered polls:      337

D execute batches:          14,987
D connector control batches:   438
D skipped empty polls:      55,935
D completion notifications:     77
D event-triggered polls:        70
D timer-triggered polls:       354
```

event=1 的 IPC 下降确实非常明显：

```text
P: 102,841 -> 443，降低 99.57%
D:  52,290 -> 438，降低 99.16%
```

但 fallback 仍是 control 的主要来源：

```text
P: 337 / 443 = 76.1%
D: 354 / 438 = 80.8%
```

notification lag：

```text
P max lag = 0.70 ms
D max lag = 23.71 ms
```

因此 completion Pipe 本身不是主要瓶颈。

### 16.5 复测后的原因分析

复测继续支持以下解释：

1. event=1 跳过了大量空 connector control，使 worker 不再每约 3ms 调用一次
   `get_finished()`；
2. 旧 `get_finished()` 会通过 `run_coroutine_threadsafe()` 向 Mooncake
   sender/receiver event loop 提交 coroutine；
3. 这些高频提交虽然带来 IPC 和 CPU 开销，但也改变了 Mooncake event loop
   的调度节奏，并为 completion 提供了一个自然的 3ms 聚合窗口；
4. event=1 改成 notification/fallback 后，P/D batch formation 和 KV readiness
   暴露时序变差；
5. 虽然 connector IPC 降低 `99%` 以上，但该收益没有转化为端到端性能。

特别注意：50ms fallback 上一轮产生了 7,000+ 次 timer control，性能比
1000ms 更差，说明问题不是简单的“兜底等待太长”或“fallback 次数太少”。

### 16.6 当前建议

1. 保持默认关闭：

    ```bash
    INFINILM_MODEL_RUNNER_CONNECTOR_COMPLETION_EVENT=0
    ```

2. 不建议继续单独扫描 10ms/20ms/50ms/100ms fallback interval；
3. 如继续优化，应先修正 fallback deadline：只有 timer-triggered control
   才推进下一次 fallback deadline，metadata/event control 不应重置它；
4. 更关键的是保留 Mooncake event loop 的调度副作用或提供 worker 内部
   debounce/heartbeat，而不是简单减少父进程 IPC；
5. event=1 若继续实验，应按 event=0/event=1 交替重复多轮，并同时统计：

    ```text
    TTFT 分布
    D KV readiness
    P send -> D ready gap
    P/D batch distribution
    connector event/timer/skip counters
    notification -> scheduler commit lag
    ```

## 17. 2026-11-04 风险修复记录

针对代码审查发现的三个问题，已完成以下修复：

### 17.1 普通 execute response 顺延 fallback deadline

普通 `execute_model()` 在 worker 侧也会调用 `get_finished()`，因此其 response
被视为一次 completion 检查，会顺延下一次 fallback deadline。这样活跃 decode
阶段不会在每次 interval 到期后额外产生空 connector control。

### 17.2 只有 timer control 重置 fallback deadline

`should_poll_connector()` 现在记录触发原因：

```text
metadata
event
timer
skip
```

connector control response 中只有 `timer` 触发的 control 会重置 fallback
deadline；metadata/event control 不重置，避免把固定兜底周期改成“任意 control
后 interval”。

### 17.3 completion Pipe 断开后停止继续入队

worker notifier 遇到 `BrokenPipeError` / `EOFError` / `OSError` 后：

```text
1. 标记 completion channel failed；
2. 清空本地 notification queue；
3. 后续 Mooncake callback 直接返回，不再向无消费者 queue 追加数据；
4. 关闭 completion write end。
```

该修复避免 parent reader 异常或 Pipe 半断开时，worker 内部无界 queue 无限
增长。
