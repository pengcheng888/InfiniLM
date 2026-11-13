# Prefill-Direct 性能报告

## 运行环境

| 项目 | 值 |
| --- | --- |
| 日期 | `<YYYY-MM-DD HH:mm-HH:mm>` |
| 平台 | `<nvidia / metax / ...>` |
| 设备 | `<device model>` |
| Driver / runtime | `<driver and runtime versions>` |
| InfiniLM build | `<commit or build ID>` |
| 模型 | `<model path>` |
| 模式 | `prefill-direct` |
| Attention backend | `<backend>` |
| Cache | paged KV cache |
| Tensor parallel size | `<TP>` |
| Input length | `<input_len>` |
| Batch size range | `<batch range>` |
| Warmup | 每个形状 3 次 |
| Measured repeats | 每个形状 `<N>` 次 |
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
BENCHMARK_MODE=prefill-direct \
BATCH_SIZES="<batch list>" \
INPUT_LENS=<input_len> \
BENCHMARK_REPEATS=<N> \
BENCHMARK_OUTPUT=<output.jsonl> \
./run_bench.sh
```

## 执行模式与 Graph 覆盖

- Graph request: `<enabled | disabled>`
- Expected execution mode: `<graph | eager>`
- Graph coverage: `<describe compiled/fallback shapes>`
- Mode source: `<compiler rule | runtime callback>`

## 资源开销

| 项目 | 值 |
| --- | ---: |
| Workspace allocation | `<value>` |
| Peak device memory | `<value>` |
| Weight loading time | `<value>` |
| Graph compilation time | `<value or not applicable>` |
| Total benchmark wall time | `<value>` |

## 测试口径

- Each sample is one physical prefill forward.
- `token_count = batch_size * input_len`.
- `throughput = token_count / avg_forward_duration_s`.
- Timing scope is `engine_forward_including_sampling`.
- Direct prefill uses synthetic input and zero-initialized paged KV cache.
- Generated tokens are not used for correctness evaluation.
- Percentiles use linear interpolation.

## 核心结果

| Batch | Input tokens | Avg ms | Min ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Throughput tok/s | Scaling |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `<b>` | `<b * input_len>` |  |  |  |  |  |  |  |  |  |

## 趋势图

将模板复制到 `docs/performance-reports/<platform>/` 后使用；下图路径基于
平台目录内的 `assets/` 目录。

### 1. Batch size 与 Avg Latency

![Batch size versus average prefill latency](assets/<run-id>-avg-latency-trend.png)

### 2. Batch size 与 Throughput

![Batch size versus prefill throughput](assets/<run-id>-throughput-trend.png)

### 3. Batch size 与 Scaling

![Batch size versus throughput scaling](assets/<run-id>-scaling-trend.png)

## 关键批次或区间

| Batch / range | Observation |
| --- | --- |
| `<batch or range>` | `<latency, throughput, or anomaly>` |

## 结果分析

1. `<latency trend>`
2. `<throughput trend and peak>`
3. `<stability and percentile behavior>`
4. `<per-request latency, if relevant>`

## 结论与建议

1. `<throughput-oriented conclusion>`
2. `<latency-oriented conclusion>`
3. `<follow-up tests>`

## 数据完整性

```text
config records: <count>
sample records: <count>
summary records: <count>
scaling records: <count>
samples per batch: <count>
```
