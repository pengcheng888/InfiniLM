# InfiniLM 普通服务 Timing 分析示例（100 请求）

测试时间：2026-11-10 14:05-14:10（Asia/Shanghai）

本文基于一次普通 InferenceServer 的 100 请求实测数据，展示 timing 数据完整性检查、请求级阶段耗时、prefill/decode batch 分布、GPU forward 空泡和瓶颈分析方法。文中数值属于该次测试的实测结果。

## 1. 测试配置

| 项目 | 值 |
|---|---|
| 服务类型 | 普通 InferenceServer，无 Mooncake PD 分离 |
| 模型 | `/data-aisoft/mechdancer/models/Qwen3-8B` |
| 服务脚本 | `run_server.sh` |
| Timing 开关 | `INFINILM_TIMING=true` |
| 服务 PID | `3399225` |
| 压测脚本 | `/workspace_v0.2.9/InfiniLM-Test/run_matrix_reqs.sh` |
| 正式请求数 | 100 |
| 并发 | 8 |
| 输入长度 | 1024 tokens |
| 输出长度 | 1024 tokens |
| Warmup | 8 个请求，每请求输出 4 tokens |
| 采样间隔 | 32 tokens |

压测命令：

```bash
TOTAL_REQUESTS=100 \
  bash /workspace_v0.2.9/InfiniLM-Test/run_matrix_reqs.sh
```

结果目录：

```text
/workspace_v0.2.9/InfiniLM-Test/benchmark_infinilm_matrix_20261110_080557
```

原始 timing 文件：

```text
timing/3399225/timing.jsonl
```

分析时剔除了脚本自带的 8 个 warmup 请求；以下所有 timing 统计只包含 100 个正式请求。

## 2. Benchmark 结果

| 指标 | 结果 |
|---|---:|
| Successful requests | 100 |
| Failed requests | 0 |
| Benchmark duration | 237.30 s |
| Request throughput | 0.42 req/s |
| Input token throughput | 431.52 tok/s |
| Output token throughput | 431.52 tok/s |
| Total token throughput | 863.03 tok/s |
| Mean E2EL | 18322.46 ms |
| Median E2EL | 18371.63 ms |
| P99 E2EL | 18698.81 ms |
| Mean TTFT | 796.44 ms |
| Median TTFT | 908.53 ms |
| P90 TTFT | 913.83 ms |
| P99 TTFT | 916.61 ms |
| Mean TPOT | 17.13 ms/token |
| Mean ITL | 17.13 ms/token |
| P99 ITL | 18.59 ms/token |

## 3. Timing 数据完整性

正式请求过滤后的 schema v3 JSONL 记录：

| Event | 数量 | 说明 |
|---|---:|---|
| `finish` | 100 | 每个正式请求 1 条 |
| `token` | 3300 | 每请求 33 个采样点 |
| `batch` | 13325 | 26 次 prefill + 13299 次 decode |
| Total | 16725 | - |

完整性校验结果：

```text
parse_error_count: 0
error_count: 0
warning_count: 0
valid: true
```

当前 schema v5 的 PD batch event 可以携带 KV connector finalize 和
`get_finished()` 时间戳。普通服务没有 KV connector，这些字段为 `null` 或缺省；
connector 相关指标不参与普通服务统计，也不应与 PD 服务比较。

数量对齐关系：

```text
token: 100 requests * 33 samples/request = 3300
finish: 100 requests = 100

prefill batch:
  batch_size=1: 13
  batch_size=3: 1
  batch_size=7: 12
  total request occurrences = 13*1 + 1*3 + 12*7 = 100

decode batch:
  batch_size=8: 12276 = 12 waves * 1023 decode steps
  batch_size=4: 1023 = 1 final wave * 1023 decode steps
  total = 13299
```

100 个请求在并发 8 下形成 12 个完整 wave 和 1 个只有 4 个请求的尾部 wave。输出 1024 tokens 中第一个 token 由 prefill 产生，后续每个请求还有 1023 次 decode forward。

## 4. 请求级阶段耗时

单位均为 ms，由 request finish event 计算。

| 阶段 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| service enter -> scheduler enqueue | 13.29 | 4.16 | 10.43 | 31.64 | 37.38 | 42.76 |
| scheduler enqueue -> prefill forward start | 94.74 | 0.62 | 105.90 | 123.01 | 123.80 | 124.22 |
| prefill forward | 672.91 | 122.50 | 769.91 | 771.76 | 771.76 | 771.76 |
| prefill complete -> loop submit | 1.07 | 0.46 | 1.09 | 1.35 | 1.35 | 1.35 |
| loop submit -> batch put start | 1.26 | 0.26 | 0.98 | 3.95 | 4.31 | 4.83 |
| first batch put | 0.18 | 0.05 | 0.16 | 0.44 | 0.44 | 0.44 |
| batch put complete -> route token received | 0.92 | 0.22 | 0.98 | 1.40 | 1.63 | 3.68 |
| route token received -> route token end | 1.34 | 0.36 | 1.37 | 2.15 | 2.43 | 2.60 |
| service enter -> first route token received | 784.37 | 132.41 | 891.98 | 910.97 | 913.13 | 913.17 |
| service enter -> final route token end | 18309.54 | 16519.41 | 18365.84 | 18683.14 | 18693.91 | 18694.98 |

请求级 forward 耗时单独拆分如下：

| 阶段 | 统计口径 | avg | min | p50 | p95 | p99 | max |
|---|---|---:|---:|---:|---:|---:|---:|
| prefill forward | 每请求一次 prefill forward 的完整耗时 | 672.91 | 122.50 | 769.91 | 771.76 | 771.76 | 771.76 |
| decode forward total | 将包含该请求的 1023 个 decode batch forward 耗时求和 | 16400.88 | 15200.90 | 16451.58 | 16624.74 | 16624.74 | 16624.74 |

说明：

- `prefill forward` 直接来自 request finish event，是每个请求参与的那一次 prefill batch forward 耗时。
- `decode forward total` 来自全部物理 decode batch event：对每个 batch，按 `request_ids` 把该 batch 的 `forward_completed_at_ns - forward_started_at_ns` 归属到其中的每个请求，再对同一请求的 1023 个 decode step 求和。该值是精确汇总，不是 token 采样估算。
- `decode forward total` 只包含 GPU forward，不包含输出队列、API event loop、SSE 传输或客户端等待。

服务端内部 first-token 平均时间为 784.37 ms，客户端 benchmark TTFT 为 796.44 ms，差值约 12.07 ms。该差值包含客户端发送、HTTP/SSE 传输和客户端侧计时边界。

服务端内部 final-token 平均时间为 18309.54 ms，客户端 mean E2EL 为 18322.46 ms，差值约 12.92 ms，与 TTFT 的客户端侧差值基本一致。

## 5. Prefill Batch 与 TTFT 分布

物理 prefill batch 分布：

| batch size | 次数 | forward avg | forward min | forward max |
|---:|---:|---:|---:|---:|
| 1 | 13 | 123.46 | 122.50 | 124.66 |
| 3 | 1 | 340.92 | 340.92 | 340.92 |
| 7 | 12 | 769.80 | 767.58 | 771.76 |

全部物理 prefill batch 的平均 forward duration 为 430.14 ms；按 request 加权后为 672.91 ms。

不同 prefill batch size 下的 forward 解释：

| batch size | 请求数 / 输入 tokens | forward avg | 平均每请求摊销 | 解释 |
|---:|---:|---:|---:|---|
| 1 | 1 / 1024 | 123.46 ms | 123.46 ms | 单请求 prefill，延迟最低，但 GPU 并行度未摊开 |
| 3 | 3 / 3072 | 340.92 ms | 113.64 ms | 尾部 wave 的 3 请求 prefill，总耗时约为 batch=1 的 2.76 倍 |
| 7 | 7 / 7168 | 769.80 ms | 109.97 ms | 完整并发 wave 中余下 7 个请求的 prefill，总耗时约为 batch=1 的 6.24 倍 |

Prefill forward 总耗时随 batch 内请求数近似线性增长：从 batch=1 到 batch=7，每增加一个请求的边际 forward 耗时约为：

```text
(769.80 ms - 123.46 ms) / (7 - 1) = 107.72 ms/request
```

因此更大的 prefill batch 能略微降低每请求摊销成本并提高 GPU 利用率，但会增加同 batch 内所有请求共同承受的 forward latency。对 TTFT 来说，batch=7 的 769.80 ms 是 median/p99 TTFT 的主要组成部分。

TTFT 分布呈明显三峰：

```text
<200 ms:     13 个请求
200-800 ms:   3 个请求
800-950 ms:  84 个请求
```

原因：

1. 并发 8 的每个完整 wave 会先调度 1 个 `batch_size=1` prefill，该请求 TTFT 约 132 ms。
2. 同 wave 其余 7 个请求随后合成 `batch_size=7` prefill，forward 约 770 ms，因此 TTFT 多落在 890-913 ms。
3. 最后一个尾部 wave 只有 4 个请求，先执行 1 个 `batch_size=1` prefill，再执行 1 个 `batch_size=3` prefill；这 3 个请求 TTFT 约 474 ms。

因此 median TTFT 和 p99 TTFT 主要由 `batch_size=7` 的 prefill 决定；如果优化 median/p99 TTFT，优先方向是首批请求到达后的 prefill 合并策略，而不是 API 输出路径。

## 6. Decode 与输出路径

Decode 物理 batch：

| batch size | 次数 | forward avg | forward p50 | forward p95 | forward p99 | forward max |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 12276 | 16.08 | 16.07 | 16.53 | 16.62 | 20.89 |
| 4 | 1023 | 14.86 | 14.86 | 15.06 | 15.10 | 16.24 |
| All | 13299 | 15.99 | 16.04 | 16.53 | 16.62 | 20.89 |

不同 decode batch size 下的 forward 解释：

- Decode 阶段每个请求每 step 只生成 1 个 token。`batch_size=8` 表示一个 forward 同时推进 8 个请求，`batch_size=4` 表示一个 forward 同时推进 4 个请求。
- `batch_size=8` 的 step forward 平均 `16.08 ms`，`batch_size=4` 平均 `14.86 ms`。batch 减半后，forward 只下降约 `1.22 ms`，降幅约 `7.6%`，说明 decode forward 中有明显的固定开销，batch 越大，每个请求的摊销成本越低。
- 从吞吐角度看，batch=8 每个 forward 产出 8 个 token，batch=4 只产出 4 个 token。折算到单位时间的理论 token 生产力：

```text
batch=8: 8 / 16.08 ms = 497.5 tokens/s
batch=4: 4 / 14.86 ms = 269.2 tokens/s
```

因此最后 4 个请求的 decode 单步延迟更低，但 GPU 利用率和总吞吐也更低。它们的 decode forward 总耗时约 15.20 s，明显低于完整 wave 请求的约 16.25-16.62 s，这也是本次 E2EL min 低于 median 约 1.85 s 的主要原因。

正式请求共有 3200 个 decode token 采样，即每请求 32 个 decode 采样点。采样到的 decode 路径分布如下：

| 阶段 | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| decode forward | 16.04 | 14.62 | 16.06 | 16.54 | 16.62 | 16.65 |
| forward complete -> loop submit | 0.72 | 0.28 | 0.72 | 1.20 | 1.32 | 1.97 |
| loop submit -> batch put start | 0.50 | 0.06 | 0.55 | 0.67 | 0.73 | 0.77 |
| batch put | 0.15 | 0.05 | 0.13 | 0.25 | 0.64 | 0.69 |
| batch put -> route received | 0.59 | 0.18 | 0.57 | 0.95 | 1.26 | 1.77 |
| route received -> route end | 1.15 | 0.32 | 1.09 | 1.86 | 2.64 | 4.10 |

客户端 mean ITL 为 17.13 ms，采样 decode forward 平均为 16.04 ms，差值约 1.09 ms。整体看，API event loop 输出路径不是当前主要瓶颈。

## 7. GPU Forward 空泡时间

本节分析物理 batch step 之间的 GPU forward 空泡。数据仍使用本次 100 个正式请求的
13,325 个物理 batch event，不包含 warmup。

### 7.1 统计口径

物理 batch event 按 `forward_started_at_ns` 排序：

```text
forward duration(i) =
    forward_completed_at_ns(i) - forward_started_at_ns(i)

gap(i) =
    forward_started_at_ns(i + 1) - forward_completed_at_ns(i)
```

其中：

- `forward duration(i)` 是第 i 个 batch step 的纯 GPU forward 耗时。
- `gap(i)` 是第 i 个 forward 结束到下一个 forward 开始之间的空泡。
- 普通服务只有一个 StepThread 串行调度物理 batch，因此相邻 forward 之间没有 GPU 计算重叠。

三个总量指标定义为：

```text
观测窗口时长 =
    last forward_completed_at_ns - first forward_started_at_ns

纯 forward 计算时间 =
    sum(forward_completed_at_ns - forward_started_at_ns)

纯 prefill forward 计算时间 =
    sum(forward_completed_at_ns - forward_started_at_ns)
    where phase == "prefill"

纯 decode forward 计算时间 =
    sum(forward_completed_at_ns - forward_started_at_ns)
    where phase == "decode"

forward 空泡时间 =
    观测窗口时长 - 纯 forward 计算时间
```

如果把“第一次 forward 开始”到“最后一次 forward 结束”称为本次 batch step 的 GPU forward
总时间，则该值对应上面的“观测窗口时长”；本节同时给出其中真正用于 forward 计算的时间和
空泡时间，避免混淆。

### 7.2 总体结果

| 指标 | 结果 |
|---|---:|
| 物理 batch forward 次数 | 13325 |
| 相邻 forward interval 数量 | 13324 |
| 第一次 forward 开始 -> 最后一次 forward 结束 | 237288.997 ms |
| 纯 forward 计算时间 | 223795.031 ms |
| 纯 prefill forward 计算时间 | 11183.546 ms |
| 纯 decode forward 计算时间 | 212611.485 ms |
| forward 空泡时间 | 13493.965 ms |
| 纯 forward 计算占比 | 94.31% |
| forward 空泡占比 | 5.69% |

按 phase 拆分纯 forward 计算时间：

| 指标 | forward 次数 | 纯 forward 计算时间 | 占观测窗口比例 |
|---|---:|---:|---:|
| 纯 prefill forward 计算时间 | 26 | 11183.546 ms | 4.71% |
| 纯 decode forward 计算时间 | 13299 | 212611.485 ms | 89.60% |
| 纯 forward 计算时间合计 | 13325 | 223795.031 ms | 94.31% |

相邻 forward 空泡分布：

| 指标 | count | sum | avg | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All intervals | 13324 | 13493.965 ms | 1.013 ms | 0.300 ms | 0.979 ms | 1.319 ms | 2.275 ms | 17.277 ms |

空泡不是由少数超长间隔贡献的。按阈值统计：

| Gap 阈值 | 数量 | 累计耗时 | 占总空泡比例 |
|---:|---:|---:|---:|
| >= 0.5 ms | 12855 | 13295.198 ms | 98.53% |
| >= 1 ms | 6019 | 7360.775 ms | 54.55% |
| >= 2 ms | 201 | 661.206 ms | 4.90% |
| >= 5 ms | 17 | 208.079 ms | 1.54% |
| >= 10 ms | 12 | 176.985 ms | 1.31% |
| >= 17 ms | 1 | 17.277 ms | 0.13% |

解释：

- 大部分空泡是每个 decode step 后约 `1 ms` 的常规调度间隔。
- `>= 2 ms` 的间隔只占空泡总量 `4.90%`，说明不存在明显的长时间 GPU 空闲段。
- 观测窗口与 benchmark duration `237.30 s` 基本一致，说明本次压测的 GPU 序列几乎覆盖正式测试全过程。

### 7.3 按前后 phase 转移拆分

| 前一个 forward | 下一个 forward | 数量 | 空泡总量 | avg | min | p50 | p95 | p99 | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode | decode | 13286 | 13239.060 ms | 0.996 ms | 0.300 ms | 0.978 ms | 1.311 ms | 2.206 ms | 7.316 ms |
| decode | prefill | 12 | 176.985 ms | 14.749 ms | 12.046 ms | 14.601 ms | 17.068 ms | 17.235 ms | 17.277 ms |
| prefill | prefill | 13 | 54.823 ms | 4.217 ms | 2.778 ms | 4.327 ms | 4.925 ms | 5.189 ms | 5.256 ms |
| prefill | decode | 13 | 23.097 ms | 1.777 ms | 1.267 ms | 1.785 ms | 1.992 ms | 2.003 ms | 2.006 ms |

各转移类型对总空泡的贡献：

| 转移类型 | 空泡总量 | 占总空泡比例 |
|---|---:|---:|
| decode -> decode | 13239.060 ms | 98.11% |
| decode -> prefill | 176.985 ms | 1.31% |
| prefill -> prefill | 54.823 ms | 0.41% |
| prefill -> decode | 23.097 ms | 0.17% |

`decode -> prefill` 的 12 个间隔对应前一个 wave 结束后，下一批请求开始 prefill 的调度边界；
其平均空泡约 `14.75 ms`，是所有转移类型中最大的，但合计只有 `176.985 ms`，占空泡总量
`1.31%`。

真正主导空泡总量的是 `decode -> decode`：13286 个连续 decode step 各自带约 `1 ms` 间隔，
累计 `13.239 s`，占空泡总量 `98.11%`。

### 7.4 空泡分析结论

1. 本次普通服务 100 请求测试中，GPU forward 序列的计算占比为 `94.31%`，空泡占比为 `5.69%`；其中纯 prefill forward 计算时间为 `11.184 s`，纯 decode forward 计算时间为 `212.611 s`。
2. 空泡主要来自高频、小间隔的 decode step 调度，而不是长时间 GPU 空闲。
3. 连续 decode forward 的平均空泡为 `0.996 ms`，p99 为 `2.206 ms`。
4. 最明显的调度边界是 `decode -> prefill`，平均 `14.749 ms`，但只出现 12 次，对整体 GPU 空泡影响很小。
5. 若要提升 GPU 利用率，优先减少每个 decode step 后约 `1 ms` 的固定间隔；等待下一个 wave 的 prefill 调度边界不是主要空泡来源。

## 8. 结论

1. **数据完整性良好**：100 个正式请求全部有 finish、首 token、按 32-token 间隔采样的 token 记录和 final token；validator 无错误。
2. **吞吐受尾部 wave 影响**：100 个请求在并发 8 下形成 12 个完整 wave 和 1 个 batch=4 的尾部 wave；尾部 wave decode 单步更快，但 GPU 利用率和吞吐更低。
3. **TTFT 瓶颈在 prefill 调度与 forward**：API 入口到 scheduler enqueue 平均仅 13.29 ms；prefill 调度等待平均 94.74 ms；request-weighted prefill forward 平均 672.91 ms。
4. **TTFT 呈明显调度双峰/三峰**：每个 wave 先有 `batch_size=1` prefill，再有余下请求的合并 prefill，导致少数请求 TTFT 约 132 ms，多数请求 TTFT 约 890-913 ms。
5. **Decode 稳定且批量摊销明显**：batch=8 decode forward 平均 16.08 ms，p99 16.62 ms；请求级 decode forward 总耗时平均 16400.88 ms；客户端 ITL 平均 17.13 ms，与采样 decode forward 平均值差约 1.09 ms。
6. **API event loop 开销较小**：decode token 的 batch put 平均 0.15 ms，route token 构造平均 1.15 ms，不是主要瓶颈。
7. **GPU forward 空泡较低**：从第一次 forward 开始到最后一次 forward 结束共 237.289 s，其中纯 forward 223.795 s，空泡 13.494 s，空泡占比 5.69%；空泡主要来自连续 decode step 间约 1 ms 的固定间隔。
8. **普通服务无 Mooncake 事件**：`kv_send` / `kv_receive` 数量为 0，`kv_connector_finalize` 指标不适用，符合无 PD 分离服务的预期。

## 9. 复现分析命令

```bash
cd /workspace_v0.2.9/InfiniLM-v0.2.9

INFINILM_TIMING=true bash run_server.sh

TOTAL_REQUESTS=100 \
  bash /workspace_v0.2.9/InfiniLM-Test/run_matrix_reqs.sh

PYTHONPATH=python python scripts/timing/validate_timing.py \
  --timing-root timing

PYTHONPATH=python python scripts/timing/compute_timing_metrics.py \
  --timing-root timing
```

注意：直接分析 `timing/` 会包含历史 PID 和 warmup 数据。报告中的正式请求统计先按本次服务 PID `3399225` 过滤，再剔除按时间排序后的前 8 个 warmup finish event。
