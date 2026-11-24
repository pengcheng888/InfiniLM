# Prefill-Direct Input 512 TP=1 性能报告

## 运行环境

| 项目 | 值 |
| ---: | ---: |
| 日期 | 2026-11-23 17:00:23-17:01:29（Asia/Shanghai） |
| 平台 | Hygon BW1000 |
| 设备 | 1 x Hygon BW1000 HCU，64 GB HBM，CUDA_VISIBLE_DEVICES 按前 1 张分配 |
| Driver / runtime | 6.3.30-V1.4.1a / DTK 26.04-rc4 |
| InfiniLM build | 分支 `pd-v0.2.9-v1-tp`，commit `da7d18e7` |
| 模型 | `/data/shared/.infinifabric-models-20261027-fef304a05440/Qwen3-8B--b968826d9c46dd6066d109eabc6255188de91218` |
| 模式 | `prefill-direct` |
| Attention backend | `flash-attn` |
| Graph request | disabled |
| Expected execution mode | `eager` |
| Cache | paged KV cache，block size 256，num blocks 16 |
| Max batched tokens | 18,432 |
| Tensor parallel size | 1 |
| Input length | 512 tokens / request |
| Batch size | 1..8 |
| Warmup / measured repeats | 每个形状 3 次 / 30 次 |
| Batch 执行顺序 | random seed 2026 |


本报告对应原始 JSONL 和 log 为本机运行产物，不随仓库提交：

```text
../../../benchmark-hygon-bw-tp1-prefill-direct-input512.jsonl
../../../benchmark-hygon-bw-tp1-prefill-direct-input512.log
```

## 构建与测试命令

```bash
xmake build _infinilm && xmake install _infinilm
```

```bash
SKIP_BUILD=1 TP=1 GPU=0 \
MODE=prefill-direct INPUT_LEN=512 \
BATCH_SIZES="$(seq -s, 1 8)" REPEATS=30 \
OUTPUT=benchmark-hygon-bw-tp1-prefill-direct-input512.jsonl ./run_bench_hygon.sh
```

## 执行模式与 Graph 覆盖

本轮 prefill-direct 不传递 `--enable-graph`，所有 physical prefill forward 走 eager path。记录中的 expected execution mode 由 benchmark 根据 compiler 规则推断，不是 C++ runtime 回传的实际执行模式。

## 资源开销

| 项目 | 值 |
| ---: | ---: |
| Workspace allocation | 每 rank 1,944 MiB，TP=1 合计 1,944 MiB |
| 权重加载时间 | 4.01 s |
| Graph compilation time | 未单独打点；包含在进程初始化到首个 sample 的 wall time 中 |
| Paged cache blocks | 16 |
| 正式 sample 数 | 240 |
| 进程 wall time | 66 s (first runtime log to final JSONL write) |
| Monitor peak HCU memory use | 35%（本 run 时间窗，分配 HCUs，10 秒采样） |
| Monitor peak package power | 415.0 W（本 run 时间窗，分配 HCUs，10 秒采样） |
| Monitor peak sensor temperature | 66.0 C（本 run 时间窗，edge/junction，10 秒采样） |


## 测试口径

- 每个 sample 是一次物理 prefill forward。
- `token_count = batch_size * 512`。
- `throughput = token_count / avg_forward_duration_s`。
- 计时边界为 `engine_forward_including_sampling`。
- Direct prefill 使用合成输入和全 0 paged KV cache，输出 token 不用于正确性判断。
- 分位数使用线性插值；每个 batch 只有 30 个正式样本，p99 只能作为尾部参考。

## 核心结果

| Batch | Input tokens | Avg ms | Min ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Throughput tok/s | Scaling |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 76.040 | 75.489 | 76.033 | 76.402 | 76.516 | 76.554 | 0.237 | 6,733.26 | 1.000x |
| 2 | 1,024 | 107.797 | 107.698 | 107.806 | 107.869 | 107.882 | 107.886 | 0.056 | 9,499.30 | 1.411x |
| 3 | 1,536 | 164.364 | 164.296 | 164.359 | 164.434 | 164.453 | 164.456 | 0.039 | 9,345.14 | 1.388x |
| 4 | 2,048 | 205.885 | 205.790 | 205.885 | 205.973 | 206.044 | 206.072 | 0.064 | 9,947.31 | 1.477x |
| 5 | 2,560 | 245.167 | 245.051 | 245.168 | 245.293 | 245.440 | 245.490 | 0.086 | 10,441.88 | 1.551x |
| 6 | 3,072 | 314.088 | 313.890 | 314.081 | 314.372 | 314.468 | 314.471 | 0.131 | 9,780.71 | 1.453x |
| 7 | 3,584 | 355.294 | 355.194 | 355.274 | 355.437 | 355.512 | 355.537 | 0.075 | 10,087.43 | 1.498x |
| 8 | 4,096 | 402.941 | 402.705 | 402.948 | 403.165 | 403.303 | 403.345 | 0.142 | 10,165.27 | 1.510x |


## 趋势图

### 1. Batch size 与 Avg Latency

![Batch size versus average prefill latency](assets/prefill-direct-hygon-bw1000-qwen3-8b-input512-batch1-8-eager-20261123-avg-latency-trend.png)

平均 latency 从 batch 1 的 76.040 ms 变为 batch 8 的 402.941 ms。

### 2. Batch size 与 Throughput

![Batch size versus prefill throughput](assets/prefill-direct-hygon-bw1000-qwen3-8b-input512-batch1-8-eager-20261123-throughput-trend.png)

吞吐峰值出现在 batch 5，为 10,441.88 tok/s。

### 3. Batch size 与 Scaling

![Batch size versus throughput scaling](assets/prefill-direct-hygon-bw1000-qwen3-8b-input512-batch1-8-eager-20261123-scaling-trend.png)

batch 8 相对 batch 1 的 scaling 为 1.551x，理想 8x 的效率约 19.4%。

## 关键批次与区间

| Batch / range | 观察 |
| ---: | ---: |
| batch 1 -> 8 | 平均 latency 从 76.040 ms 变为 402.941 ms，为 5.299x。 |
| batch 5 | 吞吐峰值 10,441.88 tok/s。 |
| batch 8 | per-request prefill latency 为 50.37 ms/request。 |


## 数据稳定性

- Std 中位数为 0.081 ms。
- 最大 std 为 0.237 ms，出现在 batch 1。
- 各 batch p99 与 max 接近，尾部稳定；每个 batch 只有 30 个 repeat，p99 是尾部参考值。

## TP 对比摘要

| TP | Best batch | Peak throughput tok/s | Scaling vs TP=1 |
| ---: | ---: | ---: | ---: |
| 1 | 5 | 10,441.88 | 1.000x |
| 2 | 5 | 16,571.70 | 1.587x |
| 4 | 5 | 24,751.22 | 2.370x |


## 分析

1. 吞吐随 batch 增加总体上升，峰值在 batch 5，为 10,441.88 tok/s。
2. 平均 forward latency 从 76.040 ms 增至 402.941 ms，增幅低于 batch 扩大倍数。
3. 吞吐峰值通常出现在最大或次大 batch，说明较大 prefill 能更好复用权重读取带宽。
4. 连续 batch 间的 latency 跳变来自算子形状/workspace 分档变化；图和完整表可定位具体区间。
5. 分位数接近均值，std 处于亚毫秒量级，本轮测量稳定性良好。

## 结论与建议

1. 该形状适合优先使用 batch 5 附近配置以获得最高 prefill 总吞吐。
2. 最大 batch 的理想 scaling 效率为 19.4%；继续扩大 batch 受权重带宽和算子效率限制。
3. 如需跨 input length 严格对比，应保持相同 `max_num_batched_tokens` 和测量轮次重跑。
4. 后续可将 eager prefill 与编译后的 prefill path 做同轮对比，定位剩余延迟来源。

## 数据完整性

```text
config records: 1
sample records: 240
summary records: 8
overall summaries: 8
scaling records: 8
samples per batch: 30
```
