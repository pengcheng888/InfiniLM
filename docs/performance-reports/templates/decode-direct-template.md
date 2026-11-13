# Decode-Direct 性能报告

## 运行环境

| 项目 | 值 |
| --- | --- |
| 日期 | `<YYYY-MM-DD HH:mm-HH:mm>` |
| 平台 | `<nvidia / metax / ...>` |
| 设备 | `<device model>` |
| Driver / runtime | `<driver and runtime versions>` |
| InfiniLM build | `<commit or build ID>` |
| 模型 | `<model path>` |
| 模式 | `decode-direct` |
| Attention backend | `<backend>` |
| Cache | paged KV cache |
| Tensor parallel size | `<TP>` |
| Input length | `<input_len>` |
| Batch size range | `<batch range>` |
| Decode steps per repeat | 4 |
| Warmup | 每个形状 3 次 repeat |
| Measured repeats | 每个形状 `<N>` 次 repeat |
| Batch 执行顺序 | random seed 2026 |

Raw artifacts are local run outputs and are not committed:

```text
<local JSONL path>
<local log path, if retained>
```

## 构建与测试命令

```bash
xmake build _infinilm && xmake install _infinilm
```

```bash
BENCHMARK_MODE=decode-direct \
BATCH_SIZES="<batch list>" \
INPUT_LENS=<input_len> \
BENCHMARK_REPEATS=<N> \
BENCHMARK_OUTPUT=<output.jsonl> \
./run_bench.sh
```

## 执行模式与 Graph 覆盖

- Graph request: `<enabled | disabled>`
- Expected execution mode: `<graph | eager | mixed>`
- Graph-covered batches: `<batch list or rule>`
- Eager-fallback batches: `<batch list or none>`
- Mode source: `<compiler rule | runtime callback>`

For mixed runs, do not interpret throughput changes across graph/eager
boundaries as a continuous batch-size effect.

## 资源开销

| 项目 | 值 |
| --- | ---: |
| Workspace allocation | `<value>` |
| Peak device memory | `<value>` |
| Weight loading time | `<value>` |
| Graph compilation time | `<value>` |
| Number of compiled graphs | `<count>` |
| Total benchmark wall time | `<value>` |

## 测试口径

- Direct decode does not perform a semantic prefill.
- The paged KV cache is zero-initialized.
- Step 0 uses a dummy token; steps 1..3 use the previous sampled token.
- Four decode steps are executed in each repeat.
- `avg_step_latency = four_step_repeat_duration / 4`.
- `single_step_throughput = batch_size / avg_step_latency`.
- One token is generated per request per step, so request throughput and token
  throughput have the same numeric value.
- Timing scope is `engine_forward_including_sampling`.
- Min/P50/P95/P99/Max in the full table describe the per-repeat average step
  time distribution unless the report explicitly aggregates individual step
  samples.

## 核心结果

| 指标 | 结果 |
| --- | ---: |
| Highest single-step throughput | `<batch>, <tok/s>` |
| Best batch average step latency | `<batch>, <ms>` |
| Scaling at largest batch | `<scaling>x` |
| Ideal-scaling efficiency | `<percentage>` |
| Median step std | `<ms>` |
| Maximum step std | `<ms>, batch <b>` |

## 趋势图

将模板复制到 `docs/performance-reports/<platform>/` 后使用；下图路径基于
平台目录内的 `assets/` 目录。

### 1. Batch size 与 Avg Step Latency

![Batch size versus average step latency](assets/<run-id>-avg-step-trend.png)

### 2. Batch size 与 Single-step Throughput

![Batch size versus single-step throughput](assets/<run-id>-throughput-trend.png)

### 3. Batch size 与 Scaling

![Batch size versus throughput scaling](assets/<run-id>-scaling-trend.png)

## 关键批次或跳变

| Change | Mode transition | Avg step ms | Throughput | Observation |
| --- | --- | ---: | ---: | --- |
| `<a> -> <b>` | `<mode>` | `<ms>` | `<tok/s>` | `<cause or hypothesis>` |

## Single-step 完整结果

Split long tables into stable ranges, for example:

### Batch `<range 1>`

| Batch | Mode | Avg Step ms | Min Step ms | P50 Step ms | P95 Step ms | P99 Step ms | Max Step ms | Std Step ms | Single-step tok/s | Scaling |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `<b>` | `<mode>` |  |  |  |  |  |  |  |  |  |

## 结果分析

1. `<throughput trend and peak>`
2. `<latency trend>`
3. `<step-to-step KV-length effect>`
4. `<graph/eager boundary or kernel-shape jumps>`
5. `<stability and p99 caveat>`

## 结论与建议

1. `<throughput-oriented conclusion>`
2. `<latency-oriented conclusion>`
3. `<graph coverage recommendation>`
4. `<follow-up tests>`

## 数据完整性

```text
config records: <count>
step samples: <count>
overall summaries: <count>
per-step summaries: <count>
scaling records: <count>
step samples per batch: <count>
samples per decode step: <count>
```
