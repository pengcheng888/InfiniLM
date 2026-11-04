# Mooncake `get_finished()` 跨线程路径优化与撤销记录

## 1. 背景

P/D 场景中，worker-side Mooncake connector 内部有两类线程：

```text
ModelRunner / connector control 调用线程
sender/receiver asyncio event loop 线程
```
旧实现中，完成事件由 sender/receiver event loop 写入普通 `set`；
`get_finished()` 被调用时，再通过
`asyncio.run_coroutine_threadsafe()` 把 fetch coroutine 调度回对应
event loop，并同步等待 `future.result()`。

这意味着每次 connector-only 空闲轮询都需要跨线程提交 coroutine：

```text
ModelRunner 线程
    -> asyncio.run_coroutine_threadsafe()
    -> sender/receiver event loop
    -> fetch_finished_*()
    -> future.result()
    -> ModelRunner 线程
```

P 侧空闲轮询频率接近 scheduler step loop 的 3ms 周期。在进程模式下，
这个路径还会与 `connector_control` IPC 叠加，产生额外 CPU 调度噪声。

## 2. 优化思路

优化目标是：完成事件生产者在 event loop 线程内只做轻量 `add/update`，
消费者在 ModelRunner 线程直接 drain，空闲轮询时不再向 asyncio loop 提交
coroutine。

核心设计是新增线程安全 completion set：

```python
class _ThreadSafeCompletionSet:
    def add(self, value):
        with self._lock:
            self._values.add(value)

    def update(self, values):
        with self._lock:
            self._values.update(values)

    def take(self):
        with self._lock:
            values = self._values
            self._values = set()
            return values
```

然后将以下三个完成集合替换为该类型：

```text
finished_sending_reqs
finished_recving_reqs
xfer_failed_recving_reqs_ids
```

优化后的 `get_finished()` 行为：

```text
D 侧:
    直接 take finished_recving_reqs
    直接 take xfer_failed_recving_reqs_ids

P 侧:
    直接 take finished_sending_reqs
    只有距离上次扫描超过 1 秒时，才向 sender loop 提交 timeout scan
```

P 侧 timeout scan 没有完全移出 sender event loop，原因是它会遍历
`reqs_need_send`，该 dict 由 sender loop 维护。保留在同一个 event loop
内可以避免并发遍历和修改同一个 dict。

## 3. 具体修改内容

### 3.1 新增线程安全 completion set

新增 `_ThreadSafeCompletionSet`，包含：

```text
add(value)
update(values)
take()
__iter__()
__contains__()
__len__()
```

其中 `take()` 是一次性 drain，返回旧集合并替换为新空集合。

### 3.2 完成事件写入

sender/receiver event loop 原有的：

```python
self.finished_sending_reqs.add(...)
self.finished_recving_reqs.update(...)
self.xfer_failed_recving_reqs_ids.update(...)
```

调用形式保持不变，但对象变成线程安全 completion set。

### 3.3 `get_finished()` 直接 drain

优化前：

```python
recv_fut = asyncio.run_coroutine_threadsafe(
    self.fetch_finished_recving_reqs(), self.receiver_loop
)
failed_recv_fut = asyncio.run_coroutine_threadsafe(
    self.fetch_xfer_failed_recving_reqs(), self.receiver_loop
)
send_fut = asyncio.run_coroutine_threadsafe(
    self.fetch_finished_sending_reqs(), self.sender_loop
)
...
recv_fut.result()
failed_recv_fut.result()
send_fut.result()
```

优化后：

```python
if not self.is_kv_producer:
    finished_recving_reqs = self.finished_recving_reqs.take()
    failed_recving_reqs = self.xfer_failed_recving_reqs_ids.take()

if not self.is_kv_consumer:
    finished_sending_reqs = self.finished_sending_reqs.take()
```

空闲时不再调度 receiver/sender coroutine。

### 3.4 P 侧 timeout scan 限频

新增：

```python
self._last_sender_timeout_poll_s = 0.0
```

只有超过 1 秒才执行：

```python
asyncio.run_coroutine_threadsafe(
    self._fetch_timeout_finished_sending_reqs(),
    self.sender_loop,
).result()
```

## 4. 与其他优化的关系

这组改动只位于：

```text
python/infinilm/kv_connector/mooncake/mooncake_connector_worker.py
```

同阶段 `mooncake_connector.py` 和
`mooncake_connector_scheduler.py` 中的改动属于另一项优化：

```text
MooncakeConnectorMetadata.is_empty()
build_connector_meta() 空时不派发 metadata
```

那部分不是 `get_finished()` 跨线程优化本身。因此本次撤销只恢复
`mooncake_connector_worker.py` 的 `get_finished()` 相关实现，保留
connector control、空 metadata 和空 output 优化，以便单独评估性能影响。

## 5. 优化后的观察结果

在 2026-11-04 TCP P/D 测试中，包含该优化在内的整体结果为：

```text
process=true + queue2 + request_cache + Pipe + busy20ms
```
结果：

```text
duration: 249.69 s
output throughput: 410.11 tok/s
mean TTFT: 1,965.51 ms
```

同请求对齐 P `kv_send` 完成时间和 D `kv_receive` 完成时间后：

```text
P send completed -> D ready completed mean = 27.13 ms
```

而优化前历史同配置为：

```text
P send completed -> D ready completed mean = 1,775.65 ms
```

需要注意：该轮提升来自多项改动叠加：

1. connector-only 专用 IPC；
2. 空 metadata 不派发；
3. 空 output 不完整序列化；
4. `get_finished()` 跨线程路径优化；
5. run-to-run 调度波动。

因此不能把全部收益单独归因给 `get_finished()`。本文记录的是当时的实现
思路；随后撤销该实现并重测，用于隔离它的影响。

## 6. 撤销范围

撤销以下内容：

1. `_ThreadSafeCompletionSet`；
2. `_last_sender_timeout_poll_s`；
3. 三个完成集合的类型替换；
4. `fetch_finished_*()` 的 direct-take 实现；
5. `_fetch_timeout_finished_sending_reqs()`；
6. `get_finished()` direct-drain 实现。

恢复为原始实现：

```text
普通 set
asyncio.run_coroutine_threadsafe(fetch_finished_*)
future.result()
```

保留以下内容：

```text
connector_control IPC
空 metadata 判定与 None 语义
空 output 编码优化
IPC 精确统计
```

## 7. 撤销后的测试目的

撤销后使用相同配置重测：

```text
TCP P/D
process=true
queue=2
request_cache=1
IPC=pipe
fixed_array=0
busy_loop=20000
100 requests
concurrency=8
input/output=1024
```

并与包含 `get_finished()` 优化的一轮比较：

```text
optimized duration: 249.69 s
optimized throughput: 410.11 tok/s
optimized mean TTFT: 1,965.51 ms
```

如果撤销后性能明显下降，则说明 direct drain 有实际收益；如果差异很小或
反向变化，则说明此前收益主要来自其他 connector control 优化或单轮调度
波动。

## 8. 撤销后复测结果

### 8.1 代码与测试状态

撤销后：

```text
mooncake_connector_worker.py == HEAD 版本
```

确认以下符号已不存在：

```text
_ThreadSafeCompletionSet
_last_sender_timeout_poll_s
_fetch_timeout_finished_sending_reqs
```

`get_finished()` 恢复为：

```text
asyncio.run_coroutine_threadsafe(fetch_finished_*)
future.result()
```

保留：

```text
connector_control IPC
MooncakeConnectorMetadata.is_empty()
空 metadata 不派发
空 output 不完整序列化
IPC 精确统计
```

验证结果：

```text
test_model_runner_process + test_mooncake_bootstrap: 29 passed
test/service --ignore test_cluster --ignore test_etcd_registry:
238 passed
python -m py_compile: passed
git diff --check: passed
```

### 8.2 Client 对比

两轮均为 `100/100` 成功，全部生成 `1024` tokens。

| 指标 | 保留 direct drain | 撤销 direct drain | 撤销后变化 |
|---|---:|---:|---:|
| Benchmark duration | 249.69 s | 252.63 s | +2.94 s，+1.18% |
| Mean TTFT | 1,965.51 ms | 1,694.20 ms | -271.31 ms，-13.81% |
| Median TTFT | 2,021.74 ms | 1,693.29 ms | -328.45 ms |
| P99 TTFT | 2,304.85 ms | 1,952.02 ms | -352.83 ms |
| Mean TPOT | 16.93 ms | 17.41 ms | +0.48 ms，+2.84% |
| Mean ITL | 67.64 ms | 69.56 ms | +1.92 ms，+2.84% |
| Mean E2EL | 19,284.97 ms | 19,505.86 ms | +220.88 ms，+1.15% |
| Output throughput | 410.11 tok/s | 405.34 tok/s | -4.77 tok/s，-1.16% |

### 8.3 Batch 与 KV timing 对比

| 指标 | 保留 direct drain | 撤销 direct drain |
|---|---:|---:|
| P prefill batch 数 | 30 | 30 |
| P prefill forward 总和 | 12,390.95 ms | 12,424.97 ms |
| P KV send mean | 1,126.12 ms | 845.11 ms |
| D decode batch 数 | 13,363 | 13,367 |
| D batch size=8 数 | 12,254 | 12,249 |
| D batch size=4 数 | 1,028 | 1,026 |
| D decode forward 总和 | 209,974.79 ms | 209,883.69 ms |
| D KV readiness mean | 1,893.80 ms | 1,625.41 ms |
| P send end -> D ready end mean | 27.13 ms | 26.86 ms |

关键观察：

1. P/D batch 分布几乎完全相同；
2. D decode forward 总和只差 `91.10 ms`；
3. P send 完成到 D ready 完成的 mean 只差 `0.27 ms`；
4. 撤销后 D KV readiness 反而降低 `268.39 ms`；
5. 撤销后 TTFT 也降低 `271.31 ms`；
6. 撤销后 TPOT/ITL 恶化约 `2.84%`，导致总吞吐下降约 `1.16%`。

### 8.4 结论

这轮隔离测试不能证明 `get_finished()` direct drain 带来了此前观察到的主要
性能收益。

更稳妥的结论是：

1. direct drain 的理论收益是减少空闲 connector-only 轮询的跨线程调度；
2. 在当前 100 请求 TCP PD workload 中，它的端到端影响只有约 `1%` 量级；
3. 撤销后 TTFT 和 D readiness 反而更好，说明单轮 TCP/调度波动不可忽略；
4. 此前 `duration 266.18 s -> 249.69 s`、吞吐提升 `6.61%` 的主要收益
   更可能来自：

   ```text
   connector_control 专用 IPC
   空 metadata 不派发
   空 output 不完整序列化
   P/D batch 形成节奏改善
   ```

5. 若要证明 direct drain 本身有稳定收益，需要按以下方式重复：

   ```text
   direct-drain / old-path / direct-drain / old-path / direct-drain / old-path
   ```

   并统计多轮均值与方差；单轮 A/B 不足以做最终结论。

### 8.5 当前建议

在得到多轮 A/B 数据前，保留当前撤销状态是合理的：

```text
使用旧的 asyncio.run_coroutine_threadsafe 路径
保留 connector_control 与空 metadata/空 output 优化
```

原因：

1. 旧路径是已验证的行为，线程所有权更清晰；
2. direct drain 的实测收益不明确；
3. 当前主要性能瓶颈在 batch formation、D readiness 与 TCP 调度，而不是
   单次 `get_finished()` 的几十至几百微秒级调度开销。
