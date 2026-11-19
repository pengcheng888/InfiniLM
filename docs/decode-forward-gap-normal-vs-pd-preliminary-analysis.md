# 普通服务与 PD Decode Forward 空泡差异初步分析

数据来源（2026-11-11 历史实验）：

```text
analysis_reports/timing-normal-service-analysis-100-20261111.md
PD 侧原始报告 timing-pd-service-analysis-100-20261111.md 已删除
```

本文记录 2026-11-11 一次 100 请求实验中观察到的现象，并给出当时的初步结论。它基于
历史 StepThread 空闲策略和旧 timing 执行模型，不是当前代码的性能基线；PD 侧当前
参考报告为 `analysis_reports/timing-pd-service-v1-100-syncstream-fixed-20261118.md`。
本文不是最终因果证明；更精确的归因需要继续增加 StepThread 内部分段时间戳或做
连接器路径消融实验。

## 1. 现象

两组历史数据中的 GPU forward 间隔平均值为：

| 服务 | 统计对象 | interval count | avg |
|---|---|---:|---:|
| 普通服务 | 全部物理 batch forward 间隔 | 13324 | 1.002 ms |
| PD Decode | Decode 进程内全部 decode batch forward 间隔 | 13742 | 1.196 ms |

直接对比会得到：

```text
1.196 ms - 1.002 ms = 0.194 ms
```

但普通服务没有独立 D 节点；它是一个进程内的 prefill/decode 混合序列。因此，严格比较
decode 行为时应先过滤普通服务中的连续 `decode -> decode` interval。

## 2. 更严格的 decode-only 对比

普通服务连续 decode interval：

```text
count: 13286
avg:   0.986 ms
sum:   13103.852 ms
```

PD Decode interval：

```text
count: 13742
avg:   1.196 ms
sum:   16437.190 ms
```

因此更严格的差异是：

```text
1.196 ms - 0.986 ms = 0.210 ms
```

分布对比：

| 序列 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| 普通服务 decode -> decode | 0.986 ms | 0.269 ms | 0.990 ms | 1.367 ms | 2.238 ms | 5.260 ms |
| PD Decode | 1.196 ms | 0.486 ms | 1.170 ms | 1.540 ms | 1.843 ms | 5.875 ms |

PD 的 p50 和 p95 都整体更高，说明这不是少数极端长尾把平均值拉高，而是分布整体右移。

## 3. 控制 batch 形态后差异仍存在

最强的一组对比是两边都只看：

```text
decode batch_size=8 -> decode batch_size=8
```

| 序列 | count | avg | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| 普通服务 batch=8 -> batch=8 | 12264 | 1.016 ms | 1.021 ms | 1.373 ms | 2.262 ms |
| PD batch=8 -> batch=8 | 11925 | 1.215 ms | 1.193 ms | 1.537 ms | 1.765 ms |

相同 batch 形态下：

```text
1.215 ms - 1.016 ms = 0.199 ms
```

这说明约 `0.2 ms` 的额外间隔不是由 batch size 分布不同偶然造成的。即使都处于稳态
batch=8 decode，PD Decode 的 host-side 间隔仍系统性更高。

## 4. 总空泡增加量分解

使用报告中的全序列总量：

| 服务 | interval count | 总空泡 |
|---|---:|---:|
| 普通服务 | 13324 | 13353.372 ms |
| PD Decode | 13742 | 16437.190 ms |

总差值：

```text
16437.190 ms - 13353.372 ms = 3083.818 ms
```

可以将差值近似分成两部分。

### 4.1 interval 数量增加

```text
13742 - 13324 = 418
```

按普通服务平均空泡 `1.002 ms` 估算：

```text
418 * 1.002 ms = 418.9 ms
```

这部分来自 PD Decode 的物理 batch 更碎、interval 数量更多。

### 4.2 单个 interval 平均变大

平均差值：

```text
1.196 ms - 1.002 ms = 0.194 ms
```

按 PD interval 数量放大：

```text
0.194 ms * 13742 = 2665 ms
```

两项相加：

```text
418.9 ms + 2665 ms = 3083.9 ms
```

因此：

```text
总空泡增加约 3.08 s
其中约 0.42 s 来自 interval 数量增加
其中约 2.67 s 来自单个 interval 平均增加
```

主导项是每个 forward 之间约 `0.2 ms` 的系统性额外开销。

## 5. 初步归因

### 5.1 PD Decode StepThread 有额外连接器控制面

GPU forward 空泡定义为：

```text
next.forward_started_at_ns
- current.forward_completed_at_ns
```

它不是 GPU 内部空洞，而是当前 forward 返回后，StepThread 完成当前输出处理、调度下一批
请求、构建下一次模型输入，直到下一次 forward 开始之前的 host-side 时间。

普通服务的 StepThread 主要执行：

```text
forward 返回
sampled tokens 转换
request 状态更新
下一轮 scheduler.schedule()
构建 model input
下一次 forward
```

PD Decode 还会持续执行 Mooncake / remote KV 相关逻辑，包括：

```text
MooncakeConnectorMetadata 构建
bind_connector_metadata()
start_load_kv()
get_finished()
get_block_ids_with_load_errors()
get_kv_connector_stats()
Scheduler.update_from_output() 中的 connector 输出处理
remote KV request / promotion 状态检查
timing events 暂存与弹出
```

相关代码路径：

- `ModelRunner.maybe_get_kv_connector_output()`
- `Scheduler.schedule()` 中每轮 `build_connector_meta()`
- `Scheduler.update_from_output()` 中 connector 状态处理

即使稳态 decode 阶段没有新的 KV 需要接收，这些 Python 对象创建、集合处理、connector
output 收集和 scheduler 状态检查仍然存在。这是当前最可能的系统性开销来源。

### 5.2 输出路径 timestamp 支持该判断

采样 decode token 中：

| 阶段 | 普通服务 avg | PD Decode avg | 差值 |
|---|---:|---:|---:|
| forward complete -> loop submit | 0.694 ms | 0.976 ms | +0.282 ms |

`forward complete -> loop submit` 是 forward 返回后 StepThread 继续处理当前输出的其中
一段。PD 明显更重，与 forward 间空泡增加约 `0.2 ms` 的方向一致。

### 5.3 `forward complete -> loop submit` 的具体代码路径

`forward_completed_at_ns` 在 `ModelRunner._model_forward()` 中、
`model_engine.forward()` 返回后立即记录。`loop_submit_started_at_ns` 则在
`AsyncLLMEngine._step_loop()` 中、`engine.step()` 返回之后记录。因此该区间包含：

```text
1. sampled tensor 转换:
   sampled_tokens.to_numpy().tolist()

2. ModelRunner connector context manager 退出:
   kv_connector.get_finished()
   kv_connector.get_block_ids_with_load_errors()
   kv_connector.get_kv_connector_stats()

3. LLMEngine.step() 后处理:
   Scheduler.update_from_output()
   Scheduler.pop_timing_events()
   TimingRecorder.record_event()
   TimingRecorder.record_batch()
   request timing 字段更新

4. LLMEngine._update_requests():
   commit computed tokens
   append sampled token
   tokenizer incremental decode
   TokenOutput construction
   Scheduler.complete_requests()

5. engine.step() 返回到 _step_loop 记录 loop_submit timestamp
```

其中最可疑的 PD-only 环节是：

```python
output.finished_sending, output.failed_recving, output.finished_recving = (
    self.kv_connector.get_finished("finished_req_ids")
)
```

该调用发生在 connector context manager 的 `finally` 中，也就是在
`forward_completed_at_ns` 之后、`loop_submit_started_at_ns` 之前。

### 5.4 `get_finished()` 的跨线程开销

在 Decode 侧，`MooncakeConnectorWorker.get_finished()` 每次都会执行：

```python
recv_fut = asyncio.run_coroutine_threadsafe(
    self.fetch_finished_recving_reqs(),
    self.receiver_loop,
)
failed_recv_fut = asyncio.run_coroutine_threadsafe(
    self.fetch_xfer_failed_recving_reqs(),
    self.receiver_loop,
)

finished_recving_reqs = recv_fut.result()
failed_recving_reqs = failed_recv_fut.result()
```

这两个 coroutine 本身只做 set swap：

```python
async def fetch_finished_recving_reqs(self):
    result = self.finished_recving_reqs
    self.finished_recving_reqs = set()
    return result
```

但执行方式是 StepThread 向 Mooncake receiver asyncio 线程提交两个 future，并阻塞等待
两个 future 完成。也就是说，实际成本不在于搬运数据，而在于：

```text
跨线程 callsoon callback 提交
receiver event loop 被唤醒
future 完成通知
StepThread 阻塞和恢复
```

每个 decode step 都要执行两次该流程。单次跨线程 future 调度通常可以是几十到一百多微秒；
两次合计约 `0.2-0.3 ms`，与当前观测到的 PD 额外 `+0.28 ms` 量级吻合。

分层数据也支持这个定位：

| 序列 | batch | token 范围 | `forward complete -> loop submit` avg |
|---|---:|---|---:|
| 普通服务 | 8 | decode sampled tokens | 0.705 ms |
| PD Decode | 8 | decode sampled tokens | 0.984 ms |
| PD Decode | 8 | 排除 token_index=0 后 | 0.984 ms |
| PD Decode | 8 | 仅 token_index=0 | 0.993 ms |

PD 的额外耗时不集中在首 decode token，也不集中在 KV receive 完成事件上；排除首 token 后
依然存在。因此这是每个 decode step 都会执行的稳态开销，与 `get_finished()` 的调用频率一致。

相对次要的 PD-only 环节包括：

```text
get_block_ids_with_load_errors()
get_kv_connector_stats()
Scheduler.update_from_output() 的 connector output 分支
```

这些主要是空集合检查、锁交换或少量 dict/list 处理，预计成本低于两次跨线程 future 调度。

2026-11-11 的 schema v5 复测已直接确认该定位。在剔除 warmup 后的 13,852 个
Decode 物理 batch 中：

| 阶段 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| `kv_connector_finalize` | 0.352 ms | 0.174 ms | 0.353 ms | 0.417 ms | 0.509 ms | 1.912 ms |
| `get_finished()` | 0.344 ms | 0.171 ms | 0.345 ms | 0.408 ms | 0.501 ms | 1.901 ms |
| finalize 剩余两个调用 | 0.008 ms | 0.003 ms | 0.008 ms | 0.010 ms | 0.014 ms | 0.048 ms |

`get_finished()` 占 finalize 总耗时的约 `97.7%`。因此，此前对两个跨线程 future
调度成本的推断成立；`get_block_ids_with_load_errors()` 与 `get_kv_connector_stats()`
合计只贡献约 `0.008 ms/step`。

### 5.5 batch 碎片化是次要但真实的因素

PD Decode batch 分布：

```text
batch=1: 294
batch=2: 101
batch=3: 323
batch=4: 737
batch=6: 1
batch=7: 315
batch=8: 11972
```

普通服务 decode batch 主要只有：

```text
batch=8: 12276
batch=4: 1023
```

PD 的小 batch 来自各请求远端 KV readiness 时间不完全对齐。调度器会先推进已经 ready 的
请求，因此产生更多物理 batch 和更多 forward interval。

但这主要解释 interval 数量增加；不能解释相同 `batch=8 -> batch=8` 形态下仍存在的
约 `0.2 ms` 平均差值。

## 6. 不支持的解释

### 6.1 不是 StepThread idle 策略导致

两份 2026-11-11 报告测试时都使用：

```text
time.sleep(0.003)
```

因此该变量在两组实验中一致。

更重要的是，持续 decode 时 `did_work=true`，不会进入 idle 分支：

```python
if not did_work:
    ...
```

所以 decode step 间约 `1.2 ms` 的 host-side 间隔与 `sleep` / `Event.wait` 空闲策略无关。

### 6.2 不是 GPU forward 本身变慢

本次数据中：

```text
普通服务 decode 物理 batch forward avg ≈ 15.98 ms
PD Decode 物理 batch forward avg ≈ 15.86 ms
```

PD 的 GPU forward 平均值并没有更高。差异主要发生在 forward 结束到下一次 forward 开始之间。

## 7. 对整体性能的影响

近似 decode 周期：

| 服务 | forward avg | gap avg | 近似周期 |
|---|---:|---:|---:|
| 普通服务 | 15.98 ms | 1.00 ms | 16.98 ms |
| PD Decode | 15.86 ms | 1.20 ms | 17.05 ms |

净差约：

```text
17.05 ms - 16.98 ms = 0.07 ms/token
```

客户端 ITL 为：

```text
普通服务 mean ITL = 17.12 ms
PD mean ITL = 17.17 ms
差值 = 0.05 ms
```

二者基本一致。PD 的 forward 间空泡多约 `0.2 ms`，但部分被略低的 decode forward 耗时抵消，
最终客户端 ITL 只高约 `0.05 ms`。

## 8. 初步结论

1. 表面差异是 PD Decode gap avg `1.196 ms`，普通服务全序列 gap avg `1.002 ms`，
   差约 `0.194 ms`。
2. 严格按连续 decode interval 对比后，差异约 `0.210 ms`。
3. 在相同 `decode batch=8 -> batch=8` 形态下，差异仍约 `0.199 ms`，说明这是系统性差异。
4. `forward complete -> loop submit` 中的主要 PD-only 耗时环节已由 schema v5 确认：
   Decode 侧每步执行的 `MooncakeConnectorWorker.get_finished()` 平均 `0.344 ms`，
   占 `kv_connector_finalize` 的约 `97.7%`。它向 receiver event loop 提交两个 future
   并阻塞等待结果；错误块查询和 stats 收集合计仅约 `0.008 ms`。
5. batch 碎片化会增加 interval 数量，贡献约 `0.42 s` 总空泡增加，但不是单个 interval
   变大的主导原因。
6. 该差异对稳态 ITL 的净影响约 `0.05 ms/token`，不是当前 PD TTFT 的主要瓶颈。
7. 当前 PD TTFT 主瓶颈仍是 D 侧 KV readiness；本次实验中 `kv_receive` readiness 平均
   `444.64 ms`。

## 9. 后续验证建议

如果要确认因果链，可以增加以下观测或实验：

1. 在 StepThread 中增加更细的分段时间戳。其中 `get_finished()` 前后时间戳已在
   schema v5 中完成，剩余优先级如下：

```text
forward_returned_at_ns
sampled_tokens_converted_at_ns
scheduler_update_completed_at_ns
next_schedule_completed_at_ns
model_input_built_at_ns
next_forward_started_at_ns
```

```text
get_load_errors_completed_at_ns
get_connector_stats_completed_at_ns
scheduler_update_completed_at_ns
update_requests_completed_at_ns
```

2. 分别统计 connector 相关路径耗时，尤其是 `get_finished()` 中两个阻塞 future：

```text
build_connector_meta
bind_connector_metadata
start_load_kv
get_finished
get_block_ids_with_load_errors
get_kv_connector_stats
```

3. 做 connector no-op 消融实验，在保证请求生命周期不变的前提下跳过稳态 decode 中的空
   connector 操作。

4. 用相同 batch size 分层比较 forward gap，优先持续观察 `batch=8 -> batch=8`。

5. 在相同 StepThread idle 策略下重复多次实验，排除单次运行噪声。
