# Prefill-Direct Input 1024 TP=4 性能报告

## 运行环境

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-11-18 15:41:44-15:42:26（Asia/Shanghai） |
| 平台 | NVIDIA |
| 设备 | 4 x A100-SXM4-80GB，默认 CUDA_VISIBLE_DEVICES=3,4,5,6 |
| Driver | 535.161.08 |
| InfiniLM build | 当前源码树；未记录独立 build ID |
| 模型 | `/data-aisoft/mechdancer/models/Qwen3-8B` |
| 模式 | `prefill-direct` |
| Attention backend | `flash-attn` |
| Graph request | enabled |
| Expected execution mode | `eager` |
| Cache | paged KV cache，block size 256，num blocks 32 |
| Max batched tokens | 18432 |
| Tensor parallel size | 4 |
| Input length | 1024 tokens / request |
| Batch size | 1..8 |
| Warmup | 每个形状 3 次 repeat |
| Measured repeats | 每个形状 30 次 repeat |
| Batch 执行顺序 | random seed 2026 |

本报告对应的原始 JSONL 和 log 为本机运行产物，不随仓库提交：

```text
../../../benchmark-prefill-direct-qwen3-8b-flash-attn-tp4-input1024-20261118.jsonl
../../../benchmark-prefill-direct-qwen3-8b-flash-attn-tp4-input1024-20261118.log
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
INPUT_LENS=1024 \
BENCHMARK_REPEATS=30 \
BENCHMARK_OUTPUT=benchmark-prefill-direct-qwen3-8b-flash-attn-tp4-input1024-20261118.jsonl \
./run_bench-nvidia.sh
```

## 执行模式与 Graph 覆盖

`run_bench-nvidia.sh` 请求开启 graph compiling，但 paged graph compiler 的 graph key 只匹配 decode-only 形状；packed prefill 形状不匹配该 key，因此本轮 expected execution mode 为 `eager`。该字段由 benchmark 根据 compiler 规则推断，不是 C++ runtime 回传的实际执行模式。

## 资源开销

| 项目 | 值 |
| --- | ---: |
| Workspace allocation | 每 rank 702 MiB，TP=4 合计 2,808 MiB |
| 权重加载时间 | 约 6.61 秒 |
| Paged cache blocks | 32 |
| 正式 sample 数 | 240 |
| 本轮总耗时 | 约 42 秒 |

Input length 1024 时，batch 8 的 prefill token 数为 `8 * 1024 = 8192`，低于 `max_num_batched_tokens=18432`，因此 workspace 上限保持 18432。

## 测试口径

- 每个 sample 是一次物理 prefill forward。
- `token_count = batch_size * 1024`。
- `throughput = token_count / avg_forward_duration_s`。
- 计时边界为 `engine_forward_including_sampling`。
- Direct prefill 使用合成输入和全 0 paged KV cache，输出 token 不用于正确性判断。
- 分位数使用线性插值；每个 batch 有 30 个正式样本，p99 只能作为尾部参考。

## 核心结果

| Batch | Input tokens | Avg ms | Min ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Throughput tok/s | Scaling |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,024 | 39.155 | 38.926 | 39.118 | 39.471 | 39.564 | 39.574 | 0.150 | 26,152.46 | 1.000x |
| 2 | 2,048 | 65.427 | 65.310 | 65.421 | 65.589 | 65.669 | 65.683 | 0.085 | 31,302.11 | 1.197x |
| 3 | 3,072 | 93.763 | 93.606 | 93.758 | 93.899 | 93.947 | 93.964 | 0.096 | 32,763.53 | 1.253x |
| 4 | 4,096 | 118.502 | 118.294 | 118.483 | 118.699 | 118.750 | 118.766 | 0.132 | 34,564.75 | 1.322x |
| 5 | 5,120 | 145.265 | 144.982 | 145.282 | 145.403 | 145.819 | 145.987 | 0.177 | 35,245.91 | 1.348x |
| 6 | 6,144 | 171.584 | 171.347 | 171.585 | 171.758 | 171.830 | 171.857 | 0.136 | 35,807.43 | 1.369x |
| 7 | 7,168 | 201.633 | 201.373 | 201.632 | 201.830 | 201.848 | 201.852 | 0.136 | 35,549.67 | 1.359x |
| 8 | 8,192 | 227.805 | 227.539 | 227.814 | 228.015 | 228.097 | 228.116 | 0.145 | 35,960.64 | 1.375x |

## 趋势图

### 1. Batch size 与 Avg Latency

![Batch size versus trend](assets/prefill-direct-nvidia-a100-tp4-qwen3-8b-input1024-batch1-8-eager-20261118-avg-latency-trend.png)

平均 latency 从 batch 1 的 39.155 ms 增至 batch 8 的 227.805 ms。

### 2. Batch size 与 Throughput

![Batch size versus throughput](assets/prefill-direct-nvidia-a100-tp4-qwen3-8b-input1024-batch1-8-eager-20261118-throughput-trend.png)

吞吐峰值出现在 batch 8，为 35,960.64 tok/s；最大 batch 8 为 35,960.64 tok/s。

### 3. Batch size 与 Scaling

![Batch size versus scaling](assets/prefill-direct-nvidia-a100-tp4-qwen3-8b-input1024-batch1-8-eager-20261118-scaling-trend.png)

Scaling 从 batch 1 的 1.000x 变为 batch 8 的 1.375x；相对理想 8x 的效率约为 17.2%。

## 关键批次与区间

| Batch / range | 观察 |
| --- | --- |
| batch 1 -> 8 | 平均 latency 从 39.155 ms 增至 227.805 ms，为 5.818x。 |
| batch 8 | 吞吐峰值，35,960.64 tok/s。 |
| batch 8 | per-request prefill latency 最低，为 28.48 ms/request。 |

## 数据稳定性

- Std 中位数为 0.136 ms。
- 最大 std 为 0.177 ms，出现在 batch 5。
- P50、P95、P99 和 max 普遍接近，数据稳定性较好。


## TP=1 / TP=2 / TP=4 对比摘要

| 配置 | 峰值 batch | 峰值吞吐 tok/s | 峰值 Avg ms | TP=4 / 该配置吞吐 |
| --- | ---: | ---: | ---: | ---: |
| TP=1 | 5 | 12,514.29 | 409.132 | 2.874x |
| TP=2 | 8 | 21,804.24 | 375.707 | 1.649x |
| TP=4 | 8 | 35,960.64 | 227.805 | 1.000x |

注意：TP=1 input 1024 参考报告使用 `max_num_batched_tokens=9216`；TP=2 和 TP=4 使用 18432，因此 TP=1 对比不是严格控制变量结果。

## 结论与建议

1. 在本轮 input length 1024、batch `1..8`、TP=4 范围内，**batch 8 吞吐最高**，约 35,960.64 tok/s。
2. batch 8 吞吐约 35,960.64 tok/s，与峰值差异约 0.00%；但单次 forward 平均耗时增加到 227.805 ms。
3. 若以单 request latency 为优先，batch 8 的 per-request latency 最低，为 28.48 ms/request。
4. 正式 p99 对比建议将 `BENCHMARK_REPEATS` 提高到 100 或以上。

## 数据完整性

```text
config records: 1
sample records: 240
overall summary records: 8
scaling records: 8
samples per batch: 30
```
