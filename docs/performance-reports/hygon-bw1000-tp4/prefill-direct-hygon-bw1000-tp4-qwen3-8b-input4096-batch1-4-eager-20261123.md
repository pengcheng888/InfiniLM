# Prefill-Direct Input 4096 TP=4 性能报告

## 运行环境

| 项目 | 值 |
| ---: | ---: |
| 日期 | 2026-11-23 19:50:29-19:51:33（Asia/Shanghai） |
| 平台 | Hygon BW1000 |
| 设备 | 4 x Hygon BW1000 HCU，64 GB HBM，CUDA_VISIBLE_DEVICES 按前 4 张分配 |
| Driver / runtime | 6.3.30-V1.4.1a / DTK 26.04-rc4 |
| InfiniLM build | 分支 `pd-v0.2.9-v1-tp`，commit `da7d18e7` |
| 模型 | `/data/shared/.infinifabric-models-20261027-fef304a05440/Qwen3-8B--b968826d9c46dd6066d109eabc6255188de91218` |
| 模式 | `prefill-direct` |
| Attention backend | `flash-attn` |
| Graph request | disabled |
| Expected execution mode | `eager` |
| Cache | paged KV cache，block size 256，num blocks 64 |
| Max batched tokens | 18,432 |
| Tensor parallel size | 4 |
| Input length | 4096 tokens / request |
| Batch size | 1..4 |
| Warmup / measured repeats | 每个形状 3 次 / 30 次 |
| Batch 执行顺序 | random seed 2026 |


本报告对应原始 JSONL 和 log 为本机运行产物，不随仓库提交：

```text
../../../benchmark-hygon-bw-tp4-prefill-direct-input4096.jsonl
../../../benchmark-hygon-bw-tp4-prefill-direct-input4096.log
```

## 构建与测试命令

```bash
xmake build _infinilm && xmake install _infinilm
```

```bash
SKIP_BUILD=1 TP=4 GPU=0,1,2,3 \
MODE=prefill-direct INPUT_LEN=4096 \
BATCH_SIZES="$(seq -s, 1 4)" REPEATS=30 \
OUTPUT=benchmark-hygon-bw-tp4-prefill-direct-input4096.jsonl ./run_bench_hygon.sh
```

## 执行模式与 Graph 覆盖

本轮 prefill-direct 不传递 `--enable-graph`，所有 physical prefill forward 走 eager path。记录中的 expected execution mode 由 benchmark 根据 compiler 规则推断，不是 C++ runtime 回传的实际执行模式。

## 资源开销

| 项目 | 值 |
| ---: | ---: |
| Workspace allocation | 每 rank 702 MiB，TP=4 合计 2,808 MiB |
| 权重加载时间 | 3.22 s |
| Graph compilation time | 未单独打点；包含在进程初始化到首个 sample 的 wall time 中 |
| Paged cache blocks | 64 |
| 正式 sample 数 | 120 |
| 进程 wall time | 65 s (first runtime log to final JSONL write) |
| Monitor peak HCU memory use | 14%（本 run 时间窗，分配 HCUs，10 秒采样） |
| Monitor peak package power | 373.0 W（本 run 时间窗，分配 HCUs，10 秒采样） |
| Monitor peak sensor temperature | 66.0 C（本 run 时间窗，edge/junction，10 秒采样） |


## 测试口径

- 每个 sample 是一次物理 prefill forward。
- `token_count = batch_size * 4096`。
- `throughput = token_count / avg_forward_duration_s`。
- 计时边界为 `engine_forward_including_sampling`。
- Direct prefill 使用合成输入和全 0 paged KV cache，输出 token 不用于正确性判断。
- 分位数使用线性插值；每个 batch 只有 30 个正式样本，p99 只能作为尾部参考。

## 核心结果

| Batch | Input tokens | Avg ms | Min ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Throughput tok/s | Scaling |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4,096 | 194.596 | 194.274 | 194.605 | 194.789 | 194.852 | 194.857 | 0.131 | 21,048.78 | 1.000x |
| 2 | 8,192 | 363.321 | 362.948 | 363.329 | 363.594 | 363.627 | 363.634 | 0.160 | 22,547.56 | 1.071x |
| 3 | 12,288 | 533.796 | 533.565 | 533.772 | 534.043 | 534.066 | 534.071 | 0.131 | 23,020.01 | 1.094x |
| 4 | 16,384 | 703.613 | 703.462 | 703.604 | 703.784 | 703.790 | 703.792 | 0.106 | 23,285.54 | 1.106x |


## 趋势图

### 1. Batch size 与 Avg Latency

![Batch size versus average prefill latency](assets/prefill-direct-hygon-bw1000-tp4-qwen3-8b-input4096-batch1-4-eager-20261123-avg-latency-trend.png)

平均 latency 从 batch 1 的 194.596 ms 变为 batch 4 的 703.613 ms。

### 2. Batch size 与 Throughput

![Batch size versus prefill throughput](assets/prefill-direct-hygon-bw1000-tp4-qwen3-8b-input4096-batch1-4-eager-20261123-throughput-trend.png)

吞吐峰值出现在 batch 4，为 23,285.54 tok/s。

### 3. Batch size 与 Scaling

![Batch size versus throughput scaling](assets/prefill-direct-hygon-bw1000-tp4-qwen3-8b-input4096-batch1-4-eager-20261123-scaling-trend.png)

batch 4 相对 batch 1 的 scaling 为 1.106x，理想 4x 的效率约 27.7%。

## 关键批次与区间

| Batch / range | 观察 |
| ---: | ---: |
| batch 1 -> 4 | 平均 latency 从 194.596 ms 变为 703.613 ms，为 3.616x。 |
| batch 4 | 吞吐峰值 23,285.54 tok/s。 |
| batch 4 | per-request prefill latency 为 175.90 ms/request。 |


## 数据稳定性

- Std 中位数为 0.131 ms。
- 最大 std 为 0.160 ms，出现在 batch 2。
- 各 batch p99 与 max 接近，尾部稳定；每个 batch 只有 30 个 repeat，p99 是尾部参考值。

## TP 对比摘要

| TP | Best batch | Peak throughput tok/s | Scaling vs TP=1 |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 8,640.70 | 1.000x |
| 2 | 4 | 14,736.60 | 1.705x |
| 4 | 4 | 23,285.54 | 2.695x |


## 分析

1. 吞吐随 batch 增加总体上升，峰值在 batch 4，为 23,285.54 tok/s。
2. 平均 forward latency 从 194.596 ms 增至 703.613 ms，增幅低于 batch 扩大倍数。
3. 吞吐峰值通常出现在最大或次大 batch，说明较大 prefill 能更好复用权重读取带宽。
4. 连续 batch 间的 latency 跳变来自算子形状/workspace 分档变化；图和完整表可定位具体区间。
5. 分位数接近均值，std 处于亚毫秒量级，本轮测量稳定性良好。

## 结论与建议

1. 该形状适合优先使用 batch 4 附近配置以获得最高 prefill 总吞吐。
2. 最大 batch 的理想 scaling 效率为 27.7%；继续扩大 batch 受权重带宽和算子效率限制。
3. 如需跨 input length 严格对比，应保持相同 `max_num_batched_tokens` 和测量轮次重跑。
4. 后续可将 eager prefill 与编译后的 prefill path 做同轮对比，定位剩余延迟来源。

## 数据完整性

```text
config records: 1
sample records: 120
summary records: 4
overall summaries: 4
scaling records: 4
samples per batch: 30
```
