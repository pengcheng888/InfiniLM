# Prefill-Direct Input 512 TP=4 性能报告

## 运行环境

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-11-18 15:31:43-15:32:09（Asia/Shanghai） |
| 平台 | NVIDIA |
| 设备 | 4 x A100-SXM4-80GB，默认 CUDA_VISIBLE_DEVICES=3,4,5,6 |
| Driver | 535.161.08 |
| InfiniLM build | 当前源码树；未记录独立 build ID |
| 模型 | `/data-aisoft/mechdancer/models/Qwen3-8B` |
| 模式 | `prefill-direct` |
| Attention backend | `flash-attn` |
| Graph request | enabled |
| Expected execution mode | `eager` |
| Cache | paged KV cache，block size 256，num blocks 16 |
| Max batched tokens | 18432 |
| Tensor parallel size | 4 |
| Input length | 512 tokens / request |
| Batch size | 1..8 |
| Warmup | 每个形状 3 次 repeat |
| Measured repeats | 每个形状 30 次 repeat |
| Batch 执行顺序 | random seed 2026 |

本报告对应的原始 JSONL 和 log 为本机运行产物，不随仓库提交：

```text
../../../benchmark-prefill-direct-qwen3-8b-flash-attn-tp4-input512-20261118.jsonl
../../../benchmark-prefill-direct-qwen3-8b-flash-attn-tp4-input512-20261118.log
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
INPUT_LENS=512 \
BENCHMARK_REPEATS=30 \
BENCHMARK_OUTPUT=benchmark-prefill-direct-qwen3-8b-flash-attn-tp4-input512-20261118.jsonl \
./run_bench-nvidia.sh
```

## 执行模式与 Graph 覆盖

`run_bench-nvidia.sh` 请求开启 graph compiling，但 paged graph compiler 的 graph key 只匹配 decode-only 形状；packed prefill 形状不匹配该 key，因此本轮 expected execution mode 为 `eager`。该字段由 benchmark 根据 compiler 规则推断，不是 C++ runtime 回传的实际执行模式。

## 资源开销

| 项目 | 值 |
| --- | ---: |
| Workspace allocation | 每 rank 702 MiB，TP=4 合计 2,808 MiB |
| 权重加载时间 | 约 6.72 秒 |
| Paged cache blocks | 16 |
| 正式 sample 数 | 240 |
| 本轮总耗时 | 约 26 秒 |

Input length 512 时，batch 8 的 prefill token 数为 `8 * 512 = 4096`，低于 `max_num_batched_tokens=18432`，因此 workspace 上限保持 18432。

## 测试口径

- 每个 sample 是一次物理 prefill forward。
- `token_count = batch_size * 512`。
- `throughput = token_count / avg_forward_duration_s`。
- 计时边界为 `engine_forward_including_sampling`。
- Direct prefill 使用合成输入和全 0 paged KV cache，输出 token 不用于正确性判断。
- 分位数使用线性插值；每个 batch 有 30 个正式样本，p99 只能作为尾部参考。

## 核心结果

| Batch | Input tokens | Avg ms | Min ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Throughput tok/s | Scaling |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 23.006 | 22.388 | 22.621 | 24.665 | 25.296 | 25.544 | 0.852 | 22,255.34 | 1.000x |
| 2 | 1,024 | 38.460 | 38.277 | 38.455 | 38.640 | 38.680 | 38.695 | 0.102 | 26,624.76 | 1.196x |
| 3 | 1,536 | 50.343 | 50.190 | 50.286 | 50.685 | 50.901 | 50.960 | 0.169 | 30,510.40 | 1.371x |
| 4 | 2,048 | 64.756 | 64.607 | 64.743 | 64.957 | 64.984 | 64.986 | 0.111 | 31,626.38 | 1.421x |
| 5 | 2,560 | 78.643 | 78.457 | 78.603 | 78.822 | 78.856 | 78.865 | 0.104 | 32,552.05 | 1.463x |
| 6 | 3,072 | 92.928 | 92.735 | 92.871 | 93.176 | 93.236 | 93.253 | 0.142 | 33,057.81 | 1.485x |
| 7 | 3,584 | 107.587 | 107.365 | 107.575 | 107.757 | 107.799 | 107.813 | 0.125 | 33,312.53 | 1.497x |
| 8 | 4,096 | 117.764 | 117.356 | 117.659 | 118.703 | 119.400 | 119.414 | 0.450 | 34,781.56 | 1.563x |

## 趋势图

### 1. Batch size 与 Avg Latency

![Batch size versus trend](assets/prefill-direct-nvidia-a100-tp4-qwen3-8b-input512-batch1-8-eager-20261118-avg-latency-trend.png)

平均 latency 从 batch 1 的 23.006 ms 增至 batch 8 的 117.764 ms。

### 2. Batch size 与 Throughput

![Batch size versus throughput](assets/prefill-direct-nvidia-a100-tp4-qwen3-8b-input512-batch1-8-eager-20261118-throughput-trend.png)

吞吐峰值出现在 batch 8，为 34,781.56 tok/s；最大 batch 8 为 34,781.56 tok/s。

### 3. Batch size 与 Scaling

![Batch size versus scaling](assets/prefill-direct-nvidia-a100-tp4-qwen3-8b-input512-batch1-8-eager-20261118-scaling-trend.png)

Scaling 从 batch 1 的 1.000x 变为 batch 8 的 1.563x；相对理想 8x 的效率约为 19.5%。

## 关键批次与区间

| Batch / range | 观察 |
| --- | --- |
| batch 1 -> 8 | 平均 latency 从 23.006 ms 增至 117.764 ms，为 5.119x。 |
| batch 8 | 吞吐峰值，34,781.56 tok/s。 |
| batch 8 | per-request prefill latency 最低，为 14.72 ms/request。 |

## 数据稳定性

- Std 中位数为 0.133 ms。
- 最大 std 为 0.852 ms，出现在 batch 1。
- P50、P95、P99 和 max 普遍接近，数据稳定性较好。


## TP=1 / TP=2 / TP=4 对比摘要

| 配置 | 峰值 batch | 峰值吞吐 tok/s | 峰值 Avg ms | TP=4 / 该配置吞吐 |
| --- | ---: | ---: | ---: | ---: |
| TP=1 | 5 | 12,489.58 | 204.971 | 2.785x |
| TP=2 | 8 | 21,452.86 | 190.930 | 1.621x |
| TP=4 | 8 | 34,781.56 | 117.764 | 1.000x |


## 结论与建议

1. 在本轮 input length 512、batch `1..8`、TP=4 范围内，**batch 8 吞吐最高**，约 34,781.56 tok/s。
2. batch 8 吞吐约 34,781.56 tok/s，与峰值差异约 0.00%；但单次 forward 平均耗时增加到 117.764 ms。
3. 若以单 request latency 为优先，batch 8 的 per-request latency 最低，为 14.72 ms/request。
4. 正式 p99 对比建议将 `BENCHMARK_REPEATS` 提高到 100 或以上。

## 数据完整性

```text
config records: 1
sample records: 240
overall summary records: 8
scaling records: 8
samples per batch: 30
```
