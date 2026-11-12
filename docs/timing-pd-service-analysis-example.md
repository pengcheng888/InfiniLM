# InfiniLM PD 分离服务 Timing 分析示例（100 请求）

测试时间：2026-11-10 18:23-18:27（Asia/Shanghai）

本文基于一次 Mooncake PD 分离服务的 100 请求实测数据，展示 PD 场景下 timing
数据完整性检查、P/D 请求级阶段耗时、prefill/decode batch 分布、Mooncake KV
传输和瓶颈分析方法。文中数值属于该次测试的实测结果。

## 1. 执行摘要

本报告基于 Mooncake PD 分离服务的 100 个正式请求 timing 数据。测试时：

- Prefill 与 Decode 分别运行在独立 InferenceServer 进程；
- Mooncake 使用 RDMA / MACA transport；
- Producer 侧 `skip_sampling` 路径已执行 GPU `syncStream()`；
- `INFINILM_TIMING=true`；
- 剔除了压测脚本自带的 8 个 warmup 请求。

核心结果：

| 指标 | 结果 |
|---|---:|
| Successful requests | 100 |
| Failed requests | 0 |
| Benchmark duration | 233.68 s |
| Output token throughput | 438.20 tok/s |
| Mean TTFT | 452.03 ms |
| Median TTFT | 526.69 ms |
| P99 TTFT | 971.27 ms |
| Mean E2EL | 18103.15 ms |
| Mean ITL | 17.25 ms/token |

本次数据中的主要瓶颈分两层：

1. **TTFT 关键路径在远端 KV readiness**：D 侧从 scheduler enqueue 到 first decode forward 平均 `409.51 ms`，其中 `kv_receive` readiness 平均 `394.10 ms`。
2. **E2E 主体是 1024-token decode**：每个请求 1024 次 decode forward 的总 GPU forward 耗时平均 `16380.55 ms`，是 E2EL 的主要组成部分。

## 2. 测试配置

| 项目 | 值 |
|---|---|
| 服务类型 | Mooncake PD 分离服务 |
| 模型 | `/data-aisoft/mechdancer/models/Qwen3-8B` |
| Prefill 脚本 | `run_prefill.sh` |
| Decode 脚本 | `run_decode.sh` |
| Proxy 脚本 | `run_proxy.sh` |
| Timing 开关 | `INFINILM_TIMING=true` |
| Prefill PID | `3749630` |
| Decode PID | `3752527` |
| Proxy PID | `3755202` |
| Mooncake 协议 | RDMA / MACA transport |
| 正式请求数 | 100 |
| 并发 | 8 |
| 输入长度 | 1024 tokens |
| 输出长度 | 1024 tokens |
| Warmup | 8 个请求，每请求输出 4 tokens |
| 采样间隔 | 32 tokens |
| 测试时 StepThread 空闲策略 | `time.sleep(0.003)` |

说明：本示例数据采集时，StepThread 空闲分支使用固定的
`time.sleep(0.003)`。当前代码已改为：

```python
self._step_wakeup.wait(timeout=0.003)
```

`add_request()`、abort 和 stop 会通过 `_notify_step_thread()` set 该 event，
因此空闲 StepThread 可以被提前唤醒；3ms timeout 仅作为无事件时的 fallback。
本文数值反映的是测试时的 `time.sleep(0.003)` 行为。

压测命令：

```bash
cd /workspace_v0.2.9/InfiniLM-Test

TOTAL_REQUESTS=100 \
HEALTH_URL=http://127.0.0.1:9104/health \
bash run_matrix_reqs.sh
```

`HEALTH_URL` 指向 Decode 后端健康检查接口；正式请求仍全部发往 PD Proxy 的 `8100` 端口。

结果目录：

```text
/workspace_v0.2.9/InfiniLM-Test/benchmark_infinilm_matrix_20261110_102329
```

原始 timing 文件：

```text
timing/3749630/timing.jsonl  # Prefill
timing/3752527/timing.jsonl  # Decode
```

## 3. Benchmark 结果

| 指标 | 结果 |
|---|---:|
| Successful requests | 100 |
| Failed requests | 0 |
| Benchmark duration | 233.68 s |
| Request throughput | 0.43 req/s |
| Input token throughput | 438.20 tok/s |
| Output token throughput | 438.20 tok/s |
| Total token throughput | 876.41 tok/s |
| Mean E2EL | 18103.15 ms |
| Median E2EL | 18214.48 ms |
| P99 E2EL | 18638.81 ms |
| Mean TTFT | 452.03 ms |
| Median TTFT | 526.69 ms |
| P90 TTFT | 837.53 ms |
| P99 TTFT | 971.27 ms |
| Mean TPOT | 17.25 ms/token |
| Mean ITL | 17.25 ms/token |
| P99 ITL | 18.78 ms/token |

## 4. Timing 数据完整性

本报告按 P/D 两套 request id 命名空间分别过滤 warmup：

```text
1. P 侧 finish event 按 service_entered_at_ns 排序，剔除前 8 个 warmup，
   得到正式 P request id 集合。
2. D 侧 finish event 按 service_entered_at_ns 排序，剔除前 8 个 warmup，
   得到正式 D request id 集合。
3. P 侧普通 batch/token/finish 使用正式 P request id 过滤。
4. P 侧 kv_send 使用正式 D request id 过滤。
5. D 侧 batch/token/finish/kv_receive 使用正式 D request id 过滤。
```

### Prefill

| Event | 数量 | 说明 |
|---|---:|---|
| `finish` | 100 | 每个正式 P 请求 1 条 |
| `token` | 100 | P 内部请求 `max_tokens=1`，每请求 1 个 token |
| `batch` | 64 | 物理 prefill batch |
| `kv_send` | 100 | 正式 D request id 对应的 KV 发送事件 |
| Total | 364 | - |

Prefill validator 结果：

```text
parse_error_count: 0
error_count: 0
warning_count: 0
valid: true
```

Prefill batch request occurrence：

```text
52*1 + 7*3 + 1*4 + 2*5 + 1*6 + 1*7 = 100
```

### Decode

| Event | 数量 | 说明 |
|---|---:|---|
| `finish` | 100 | 每个正式 D request 1 条 |
| `token` | 3300 | 每请求 33 个采样点 |
| `batch` | 13601 | 物理 decode batch |
| `kv_receive` | 100 | 每请求 1 条 KV readiness event |
| Total | 17101 | - |

Decode validator 结果：

```text
parse_error_count: 0
error_count: 0
warning_count: 0
valid: true
```

Decode request-step 对齐：

```text
100 requests * 1024 decode steps = 102400 request-steps
13601 physical decode batches 展开后 = 102400 request-steps
```

因此每个请求都完整经历了 1024 次 decode step，没有缺失采样或中途掉队请求。

## 5. Prefill 侧请求级耗时

单位均为 ms。

| 阶段 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| service enter -> scheduler enqueue | 8.37 | 4.72 | 6.90 | 15.79 | 20.26 | 24.11 |
| scheduler enqueue -> prefill forward start | 53.03 | 0.57 | 1.73 | 120.63 | 123.85 | 124.02 |
| prefill forward | 305.25 | 122.89 | 125.55 | 768.73 | 768.73 | 768.73 |
| prefill complete -> loop submit | 1.03 | 0.47 | 0.99 | 1.37 | 1.39 | 2.68 |
| service -> first route token | 370.12 | 131.28 | 138.30 | 896.20 | 896.78 | 910.06 |
| service -> final route token | 371.37 | 131.64 | 138.78 | 898.85 | 901.44 | 911.20 |

P 侧内部请求只生成一个 prefill token，因此 first token 与 final token 只差约 `1.25 ms`。

请求级 prefill forward 分布显示明显的 batch size 影响：

```text
所有 100 个请求的 prefill forward 均 >= 100 ms
batch_size=1 的 full prompt prefill 约 123-126 ms
```

## 6. Prefill Batch 与 Forward

| batch size | 次数 | request occurrence | forward avg | forward min | forward max |
|---:|---:|---:|---:|---:|---:|
| 1 | 52 | 52 | 124.56 | 122.89 | 125.70 |
| 3 | 7 | 21 | 341.00 | 340.02 | 342.16 |
| 4 | 1 | 4 | 454.08 | 454.08 | 454.08 |
| 5 | 2 | 10 | 571.02 | 570.06 | 571.98 |
| 6 | 1 | 6 | 663.19 | 663.19 | 663.19 |
| 7 | 1 | 7 | 768.73 | 768.73 | 768.73 |

解释：

- Proxy 将 prefill 请求独立发送给 P 服务，因此 P 侧不会稳定形成并发 8 的统一 prefill batch。
- `batch_size=1` 是主要形态，共 52 次。
- forward 总耗时随 batch 内请求数近似线性增长：batch=1 约 `124.56 ms`，batch=7 约 `768.73 ms`。
- 大 batch 提高 GPU 利用率，但同 batch 内所有请求共同承受更长的 prefill forward latency。

## 7. KV Connector Finalize

当前 timing schema v5 在物理 batch event 中增加：

```text
kv_connector_finalize_started_at_ns
kv_connector_get_finished_started_at_ns
kv_connector_get_finished_completed_at_ns
kv_connector_finalize_completed_at_ns
```

本节主表数值来自 schema v4 PD 服务示例数据；schema v5 额外拆出 `get_finished()`
耗时，文末补充了 v5 复测结果。

这两个字段包住 `ModelRunner.maybe_get_kv_connector_output()` 的 `finally`
收尾阶段，覆盖：

```text
get_finished()
get_block_ids_with_load_errors()
get_kv_connector_stats()
```

因此：

```text
kv_connector_finalize_ms =
    kv_connector_finalize_completed_at_ns
    - kv_connector_finalize_started_at_ns
```

该指标表示三个 connector 收尾调用的整体耗时，不包含后续
`Scheduler.update_from_output()`、request token update 或 API loop callback。

### D 侧采样 decode 分布

| 阶段 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| forward complete -> finalize start | 0.123 ms | 0.063 ms | 0.121 ms | 0.160 ms | 0.197 ms | 0.288 ms |
| `kv_connector_finalize` | 0.343 ms | 0.229 ms | 0.341 ms | 0.409 ms | 0.519 ms | 2.205 ms |
| finalize -> loop submit | 0.555 ms | 0.171 ms | 0.560 ms | 0.809 ms | 0.936 ms | 1.017 ms |
| forward complete -> loop submit | 1.021 ms | 0.565 ms | 1.021 ms | 1.289 ms | 1.448 ms | 2.696 ms |

解释：

- `kv_connector_finalize` 占 D 侧采样 `forward complete -> loop submit` 平均耗时的约 `33.6%`。
- 首 token 与后续 token 的 finalize 平均值接近，说明这是每个 decode step 的稳态控制面开销。
- finalize 后续处理平均 `0.555 ms`，因此不能把 post-forward 额外耗时全部归因于 connector 收尾调用。

### D 侧物理 batch 分布

| batch size | count | finalize avg | finalize p50 | finalize p95 | finalize p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 103 | 0.336 ms | 0.333 ms | 0.373 ms | 0.730 ms |
| 2 | 99 | 0.321 ms | 0.300 ms | 0.380 ms | 0.893 ms |
| 3 | 104 | 0.296 ms | 0.281 ms | 0.342 ms | 0.411 ms |
| 4 | 820 | 0.343 ms | 0.338 ms | 0.406 ms | 0.438 ms |
| 5 | 146 | 0.362 ms | 0.367 ms | 0.412 ms | 0.578 ms |
| 7 | 527 | 0.348 ms | 0.351 ms | 0.420 ms | 0.512 ms |
| 8 | 11761 | 0.342 ms | 0.342 ms | 0.405 ms | 0.482 ms |

不同 batch size 下该耗时基本稳定，说明它主要来自固定调用开销，而不是随 batch 内
request 数线性增长。

### Schema v5：`get_finished()` 拆分

2026-11-11 的 schema v5 复测剔除 warmup 后，D 侧共有 13,852 个 decode 物理 batch。
物理 batch 分布如下：

| 阶段 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| `kv_connector_finalize` | 0.352 ms | 0.174 ms | 0.353 ms | 0.417 ms | 0.509 ms | 1.912 ms |
| `kv_connector_get_finished` | 0.344 ms | 0.171 ms | 0.345 ms | 0.408 ms | 0.501 ms | 1.901 ms |
| finalize 剩余部分 | 0.008 ms | 0.003 ms | 0.008 ms | 0.010 ms | 0.014 ms | 0.048 ms |

`get_finished()` 占 `kv_connector_finalize` 的约 `97.7%`。这说明 finalize 中的固定
控制面开销主要来自 `MooncakeConnectorWorker.get_finished()` 的跨线程 future 调度；
错误块查询和 stats 收集合计不到 `0.01 ms/step`。

## 8. Mooncake KV Transfer

### P 侧 kv_send

`kv_send` 统计：

```text
request_event_count: 100
unique_batch_count: 82
```

物理 Mooncake send batch：

| batch size | 次数 |
|---:|---:|
| 1 | 68 |
| 2 | 11 |
| 3 | 2 |
| 4 | 1 |

P 侧时间点分布：

| 指标 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| KV ready -> `batch_transfer_sync_write` start | 1.81 | 0.40 | 1.53 | 3.17 | 7.19 | 7.57 |
| write duration, request-weighted | 6.95 | 3.13 | 3.81 | 14.78 | 17.71 | 17.71 |
| write duration, batch-weighted | 5.82 | 3.13 | 3.60 | 13.14 | 15.33 | 17.71 |

KV 数据量：

```text
每个请求: 4 blocks
每个请求: 150,994,944 bytes
总计: 15,099,494,400 bytes
```

`batch_transfer_sync_write` 是 Mooncake 数据面传输 API 的直接耗时；`KV ready -> write start` 包含 sender loop 唤醒、参数构建和 executor 排队。

### D 侧 kv_receive

| 指标 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| KV pull readiness | 394.10 | 115.25 | 455.97 | 893.54 | 901.92 | 909.66 |
| KV ready -> first decode forward | 8.70 | 2.41 | 7.88 | 16.27 | 17.25 | 17.44 |

`kv_cache_pull_readiness_ms` 从 D 决定拉取 KV 开始计时，包含：

```text
bootstrap / agent 查询
ZMQ control plane
等待 P 侧 prefill 完成
Mooncake KV 传输确认
```

因此它是 D 本地 KV readiness 总耗时，不是纯 RDMA 传输时间。

P/D 属于不同进程和不同 local monotonic clock domain，本报告不做跨节点 timestamp 相减。

## 9. Decode 侧请求级耗时

| 阶段 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| service enter -> scheduler enqueue | 8.29 | 4.31 | 6.30 | 16.49 | 21.70 | 24.07 |
| scheduler enqueue -> first decode forward | 409.51 | 136.16 | 473.86 | 902.89 | 915.60 | 918.65 |
| first decode forward | 15.82 | 14.01 | 15.81 | 16.34 | 16.42 | 16.44 |
| first forward complete -> loop submit | 1.01 | 0.52 | 0.98 | 1.29 | 1.58 | 2.80 |
| loop submit -> batch put start | 0.61 | 0.14 | 0.60 | 0.75 | 0.90 | 1.15 |
| first batch put | 0.14 | 0.04 | 0.14 | 0.22 | 0.30 | 0.75 |
| batch put -> route token received | 0.58 | 0.08 | 0.56 | 0.86 | 1.09 | 23.72 |
| route token received -> route token end | 0.89 | 0.10 | 0.84 | 1.46 | 1.76 | 2.95 |
| service -> first route token | 436.15 | 161.28 | 501.37 | 941.18 | 941.57 | 941.74 |
| service -> final route token | 18087.51 | 16788.76 | 18189.43 | 18598.64 | 18622.80 | 18628.74 |

客户端 mean TTFT 为 `452.03 ms`，D 服务内部 first route token 为 `436.15 ms`，差值约 `15.88 ms`。该差值主要来自 Proxy、HTTP/SSE 传输和客户端侧计时边界。

客户端 mean E2EL 为 `18103.15 ms`，D 服务内部 final route token 为 `18087.51 ms`，差值约 `15.64 ms`。

## 10. Decode Forward 与 Batch

请求级 decode forward 总耗时：

| 阶段 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| Decode forward total, 1024 steps/request | 16380.55 | 15487.59 | 16452.61 | 16538.30 | 16538.30 | 16538.30 |

物理 decode batch 分布：

| batch size | 次数 | request occurrence | forward avg | forward p50 | forward p95 | forward p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 104 | 104 | 13.36 | 13.28 | 13.63 | 14.01 |
| 2 | 188 | 376 | 14.40 | 14.44 | 14.59 | 14.87 |
| 3 | 103 | 309 | 14.71 | 14.78 | 14.86 | 15.02 |
| 4 | 918 | 3672 | 14.98 | 14.97 | 15.19 | 15.32 |
| 6 | 1 | 6 | 15.82 | 15.82 | 15.82 | 15.82 |
| 7 | 363 | 2541 | 16.00 | 15.99 | 16.22 | 16.47 |
| 8 | 11924 | 95392 | 16.05 | 16.06 | 16.39 | 16.55 |
| All | 13601 | 102400 | 15.92 | 16.02 | 16.38 | 16.55 |

Decode batch=8 占比：

```text
物理 batch: 11924 / 13601 = 87.67%
request-step: 95392 / 102400 = 93.16%
```

解释：

- D 侧需要等待远端 KV readiness，请求 ready 时间不完全对齐，因此出现 batch=1、2、3、4、6、7、8 多种形态。
- 稳态后 decode 基本回到 batch=8。
- decode forward 的批量摊销明显：batch=1 平均 `13.36 ms`，batch=8 平均 `16.05 ms`；batch size 增加到 8 倍，step forward 只增加约 `2.69 ms`。

## 11. Decode 输出路径

3300 个 decode token 采样点的输出路径：

| 阶段 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| decode forward | 16.00 | 13.22 | 16.04 | 16.39 | 16.54 | 18.51 |
| forward complete -> loop submit | 1.01 | 0.52 | 0.98 | 1.29 | 1.58 | 2.80 |
| loop submit -> batch put start | 0.60 | 0.06 | 0.60 | 0.73 | 0.81 | 5.42 |
| batch put | 0.16 | 0.01 | 0.16 | 0.23 | 0.30 | 0.75 |
| batch put -> route received | 0.58 | 0.08 | 0.56 | 0.86 | 1.09 | 23.72 |
| route received -> route end | 0.89 | 0.10 | 0.84 | 1.46 | 1.76 | 2.95 |

客户端 mean ITL 为 `17.25 ms`，采样 decode forward 平均为 `16.00 ms`，差值约 `1.25 ms`。输出队列、API event loop 和 SSE route 构造不是当前主要瓶颈。

## 12. GPU Forward 空泡时间

PD 分离服务有两个独立执行序列：

1. Prefill 进程中的物理 prefill batch forward；
2. Decode 进程中的物理 decode batch forward。

因此，不能像普通服务那样把所有 batch event 合并成一个 forward 序列。本节分别使用 P/D
各自进程内的 `time.monotonic_ns()` 时间戳计算空泡，不跨进程相减，也不计算两个进程之间的
时间重叠。

### 12.1 统计口径

每个进程内，物理 batch event 按 `forward_started_at_ns` 排序：

```text
forward duration(i) =
    forward_completed_at_ns(i) - forward_started_at_ns(i)

gap(i) =
    forward_started_at_ns(i + 1) - forward_completed_at_ns(i)
```

三个总量指标为：

```text
观测窗口时长 =
    last forward_completed_at_ns - first forward_started_at_ns

纯 forward 计算时间 =
    sum(forward_completed_at_ns - forward_started_at_ns)

forward 空泡时间 =
    观测窗口时长 - 纯 forward 计算时间
```

如果将“第一次 forward 开始”到“最后一次 forward 结束”称为该 GPU 的 forward 观测窗口，
则该窗口同时包含纯 forward 计算和 forward 间空泡。

### 12.2 Prefill GPU

Prefill 侧只有 64 个物理 prefill batch forward：

| 指标 | 结果 |
|---|---:|
| 物理 forward 次数 | 64 |
| 相邻 forward interval 数量 | 63 |
| 第一次 forward 开始 -> 最后一次 forward 结束 | 216997.144 ms |
| 纯 forward 计算时间 | 11892.047 ms |
| forward 空泡时间 | 205105.097 ms |
| 纯 forward 计算占比 | 5.48% |
| forward 空泡占比 | 94.52% |

Prefill forward 间隔分布：

| count | sum | avg | min | p50 | p90 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 63 | 205105.097 ms | 3255.636 ms | 1.767 ms | 863.889 ms | 13204.794 ms | 14644.418 ms | 16000.630 ms | 16545.665 ms |

为解释空泡来源，将 interval 分为三类：

| 转移类型 | 数量 | 空泡总量 | 占总空泡比例 | avg | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch=1 -> batch=1 | 39 | 40709.069 ms | 19.85% | 1043.822 ms | 389.210 ms | 863.889 ms | 2034.149 ms | 2829.528 ms |
| batch=1 -> grouped batch | 12 | 40.086 ms | 0.02% | 3.341 ms | 1.767 ms | 3.288 ms | 5.295 ms | 5.758 ms |
| grouped batch -> 下一 wave 的 batch=1 | 12 | 164355.942 ms | 80.13% | 13696.329 ms | 11388.840 ms | 13506.300 ms | 16062.166 ms | 16545.665 ms |

解释：

1. Prefill GPU 的空泡占比很高，但这是由角色和负载形态共同决定的：P 侧只执行 prefill，
   不执行 1024-token decode，因此它的计算量天然远小于 D 侧。
2. 最大贡献来自上一个 wave 的 grouped prefill 结束到下一个 wave 首个 `batch=1` prefill
   开始，12 个间隔合计 `164.356 s`，占 P 侧空泡 `80.13%`。
3. 同一 wave 内连续的 `batch=1 -> batch=1` 间隔也较明显，39 个间隔合计 `40.709 s`，
   占 `19.85%`。
4. `batch=1 -> grouped batch` 的调度间隔只有 `1.8-5.8 ms`，说明一旦请求已在 P 侧队列中，
   前后两个 prefill batch 的调度间隔很小。

因此，P 侧 GPU 空泡主要来自请求到达节奏和 wave 间等待，而不是相邻 forward 的 StepThread
调度开销。

### 12.3 Decode GPU

Decode 侧有 13601 个物理 decode batch forward：

| 指标 | 结果 |
|---|---:|
| 物理 forward 次数 | 13601 |
| 相邻 forward interval 数量 | 13600 |
| 第一次 forward 开始 -> 最后一次 forward 结束 | 233514.055 ms |
| 纯 forward 计算时间 | 216557.393 ms |
| forward 空泡时间 | 16956.662 ms |
| 纯 forward 计算占比 | 92.74% |
| forward 空泡占比 | 7.26% |

Decode forward 间隔分布：

| count | sum | avg | min | p50 | p90 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13600 | 16956.662 ms | 1.247 ms | 0.516 ms | 1.222 ms | 1.478 ms | 1.570 ms | 1.930 ms | 4.747 ms |

阈值分布：

| Gap 阈值 | 数量 | 累计耗时 | 占总空泡比例 |
|---:|---:|---:|---:|
| >= 0.5 ms | 13600 | 16956.662 ms | 100.00% |
| >= 1 ms | 12608 | 16068.801 ms | 94.76% |
| >= 2 ms | 121 | 292.783 ms | 1.73% |
| >= 5 ms | 0 | 0 ms | 0.00% |

解释：

1. Decode GPU 的纯 forward 计算占比为 `92.74%`，空泡占比为 `7.26%`。
2. 空泡主要来自每个 decode step 后约 `1.2 ms` 的常规调度间隔。
3. 没有任何间隔达到 `5 ms`，说明 D 侧没有明显的长时间 GPU idle 段。
4. 客户端 mean ITL 为 `17.25 ms`，decode forward 平均为 `15.92 ms`，forward 间平均空泡
   `1.25 ms`，三者与整体 decode 周期一致。

### 12.4 PD 空泡结论

1. **D 侧 GPU 利用率高**：forward 计算占比 `92.74%`，空泡占比 `7.26%`，且没有 `>=5 ms`
   的长空泡。
2. **D 侧空泡是高频小间隔**：13600 个 interval 平均 `1.247 ms`，p99 `1.930 ms`。
3. **P 侧 forward 占比低**：计算占比 `5.48%`，空泡占比 `94.52%`。
4. **P 侧低占比主要由负载形态决定**：P 只执行 prefill，且请求按 wave 到达；12 个 wave 间
   间隔贡献了 `80.13%` 的 P 侧空泡。
5. **不能用 P/D 观测窗口直接相减或合并计算总 GPU 利用率**；两者使用独立进程时钟，且是
   两个独立 GPU 序列。

## 13. 结论

1. **数据完整性通过**：P 侧 364 条、D 侧 17101 条正式 timing record，validator 均无错误；decode request-step 与 1024 输出完全对齐。
2. **TTFT 主要由 KV readiness 决定**：D 侧 `scheduler enqueue -> first decode forward` 平均 `409.51 ms`，`kv_receive` readiness 平均 `394.10 ms`。
3. **KV ready 后调度开销小**：D 侧 KV ready 到 first decode forward 平均 `8.70 ms`，p99 `17.25 ms`。
4. **Mooncake 数据面传输耗时小**：P 侧 request-weighted `batch_transfer_sync_write` 平均 `6.95 ms`，p99 `17.71 ms`。
5. **P 侧 prefill batch 碎片化明显**：batch=1 占 52 次，batch size 分布为 1、3、4、5、6、7；request-weighted prefill forward 平均 `305.25 ms`。
6. **E2E 主体是 decode forward**：每请求 1024 次 decode forward 总耗时平均 `16380.55 ms`。
7. **Decode 稳态批量较好**：batch=8 覆盖 `93.16%` request-step，batch=8 forward 平均 `16.05 ms`。
8. **输出路径开销小**：客户端 ITL 与采样 decode forward 差值约 `1.25 ms`。
9. **KV connector finalize 是稳定控制面开销**：schema v4 示例中 D 侧平均约 `0.343 ms`；schema v5 进一步确认其中约 `97.7%` 来自 `get_finished()`，错误块查询和 stats 收集合计约 `0.008 ms/step`。
10. **GPU 空泡形态分离明显**：Decode GPU forward 占比 `92.74%`，空泡来自高频约 `1.25 ms` 的 step 间隔；Prefill GPU forward 占比 `5.48%`，空泡主要来自 wave 间请求到达等待。
11. **PD 优化优先级**：优先减少 D 侧 KV readiness 等待与 P 侧 prefill batch 碎片化；随后再拆分 finalize 后续的 Scheduler/request 更新路径。输出路径和 Mooncake 数据面写调用不是第一优先级瓶颈。
