# Prefill-Direct Input 2048 TP=2 性能报告

## 运行环境

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-11-18 15:07:53-15:09:52（Asia/Shanghai） |
| 平台 | NVIDIA |
| 设备 | 2 x A100-SXM4-80GB，默认 CUDA_VISIBLE_DEVICES=3,4 |
| Driver | 535.161.08 |
| InfiniLM build | 当前源码树；未记录独立 build ID |
| 模型 | `/data-aisoft/mechdancer/models/Qwen3-8B` |
| 模式 | `prefill-direct` |
| Attention backend | `flash-attn` |
| Graph request | enabled |
| Expected execution mode | `eager` |
| Cache | paged KV cache，block size 256，num blocks 64 |
| Max batched tokens | 18432 |
| Tensor parallel size | 2 |
| Input length | 2048 tokens / request |
| Batch size | 1..8 |
| Warmup | 每个形状 3 次 repeat |
| Measured repeats | 每个形状 30 次 repeat |
| Batch 执行顺序 | random seed 2026 |

本报告对应的原始 JSONL 和 log 为本机运行产物，不随仓库提交：

```text
../../../benchmark-prefill-direct-qwen3-8b-flash-attn-tp2-input2048-20261118.jsonl
../../../benchmark-prefill-direct-qwen3-8b-flash-attn-tp2-input2048-20261118.log
```

## 构建与测试命令

本轮使用当前已安装的 `_infinilm` 扩展。标准重建命令如下：

```bash
xmake build _infinilm && xmake install _infinilm
```

性能测试命令：

```bash
BENCHMARK_MODE=prefill-direct \
BATCH_SIZES="1,2,3,4,5,6,7,8" \
INPUT_LENS=2048 \
BENCHMARK_REPEATS=30 \
BENCHMARK_OUTPUT=benchmark-prefill-direct-qwen3-8b-flash-attn-tp2-input2048-20261118.jsonl \
./run_bench-nvidia.sh
```

## 执行模式与 Graph 覆盖

`run_bench-nvidia.sh` 请求开启 graph compiling，但 paged graph compiler 的 graph key 只匹配 decode-only 形状。packed prefill 形状不匹配该 key，因此 本轮 expected execution mode 为 `eager`。该字段由 benchmark 根据 compiler 规则推断，不是 C++ runtime 回传的实际执行模式。

## 资源开销

| 项目 | 值 |
| --- | ---: |
| Workspace allocation | 每 rank 1,116 MiB，TP=2 合计 2,232 MiB |
| 权重加载时间 | 约 6.30 秒 |
| Paged cache blocks | 64 |
| 正式 sample 数 | 240 |
| 本轮总耗时 | 约 119 秒 |

Input length 2048 时，batch 8 的 prefill token 数为 `8 * 2048 = 16384`，低于 `max_num_batched_tokens=18432`，因此 workspace 上限保持 18432。

## 测试口径

- 每个 sample 是一次物理 prefill forward。
- `token_count = batch_size * 2048`。
- `throughput = token_count / avg_forward_duration_s`。
- 计时边界为 `engine_forward_including_sampling`。
- Direct prefill 使用合成输入和全 0 paged KV cache，输出 token 不用于正确性判断。
- 分位数使用线性插值；每个 batch 有 30 个正式样本，p99 只能作为尾部参考。

## 核心结果

| Batch | Input tokens | Avg ms | Min ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Throughput tok/s | Scaling |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,048 | 106.178 | 105.979 | 106.145 | 106.433 | 106.481 | 106.485 | 0.124 | 19,288.36 | 1.000x |
| 2 | 4,096 | 196.243 | 196.088 | 196.228 | 196.436 | 196.490 | 196.506 | 0.110 | 20,872.04 | 1.082x |
| 3 | 6,144 | 293.031 | 292.806 | 293.046 | 293.180 | 293.208 | 293.213 | 0.106 | 20,967.09 | 1.087x |
| 4 | 8,192 | 382.443 | 382.204 | 382.388 | 382.732 | 382.837 | 382.878 | 0.169 | 21,420.21 | 1.111x |
| 5 | 10,240 | 468.988 | 468.598 | 469.022 | 469.175 | 469.262 | 469.296 | 0.128 | 21,834.27 | 1.132x |
| 6 | 12,288 | 562.676 | 562.244 | 562.701 | 562.896 | 562.955 | 562.969 | 0.174 | 21,838.50 | 1.132x |
| 7 | 14,336 | 654.124 | 653.527 | 654.114 | 654.719 | 655.451 | 655.722 | 0.465 | 21,916.35 | 1.136x |
| 8 | 16,384 | 744.047 | 743.713 | 744.038 | 744.358 | 744.398 | 744.405 | 0.162 | 22,020.10 | 1.142x |


## 趋势图

### 1. Batch size 与 Avg Latency

![Batch size versus average prefill latency](assets/prefill-direct-nvidia-a100-tp2-qwen3-8b-input2048-batch1-8-eager-20261118-avg-latency-trend.png)

平均 latency 从 batch 1 的 106.178 ms 增至 batch 8 的 744.047 ms。

### 2. Batch size 与 Throughput

![Batch size versus prefill throughput](assets/prefill-direct-nvidia-a100-tp2-qwen3-8b-input2048-batch1-8-eager-20261118-throughput-trend.png)

吞吐峰值出现在 batch 8，为 22,020.10 tok/s；最大 batch 8 为 22,020.10 tok/s。

### 3. Batch size 与 Scaling

![Batch size versus throughput scaling](assets/prefill-direct-nvidia-a100-tp2-qwen3-8b-input2048-batch1-8-eager-20261118-scaling-trend.png)

Scaling 从 batch 1 的 1.000x 变为 batch 8 的 1.142x；相对理想 8x 的效率约为 14.3%。

## 关键批次与区间

| Batch / range | 观察 |
| --- | --- |
| batch 1 -> 8 | 平均 latency 从 106.178 ms 增至 744.047 ms，为 7.008x。 |
| batch 8 | 吞吐峰值，22,020.10 tok/s。 |
| batch 8 | per-request prefill latency 最低，为 93.01 ms/request。 |

## 数据稳定性

- Std 中位数为 0.145 ms。
- 最大 std 为 0.465 ms，出现在 batch 7。
- P50、P95、P99 和 max 普遍接近，数据稳定性较好。

## TP=1 对比摘要

| 配置 | 峰值 batch | 峰值吞吐 tok/s | 峰值 Avg ms |
| --- | ---: | ---: | ---: |
| TP=1 | 7 | 12,322.39 | 1163.411 |
| TP=2 | 8 | 22,020.10 | 744.047 |

以各自峰值吞吐对比，TP=2 吞吐为 TP=1 的 1.787x；峰值处 latency 为 TP=1 的 0.640x。

## 结论与建议

1. 在本轮 input length 2048、batch `1..8`、TP=2 范围内，**batch 8 吞吐最高**，约 22,020.10 tok/s。
2. batch 8 吞吐约 22,020.10 tok/s，与峰值差异约 0.00%；但单次 forward 平均耗时增加到 744.047 ms。
3. 若以单 request latency 为优先，batch 8 的 per-request latency 最低，为 93.01 ms/request。
4. 正式 p99 对比建议将 `BENCHMARK_REPEATS` 提高到 100 或以上。

## 数据完整性

```text
config records: 1
sample records: 240
overall summary records: 8
scaling records: 8
samples per batch: 30
```
