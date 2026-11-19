# Prefill-Direct Input 2048 TP=2 性能报告

## 运行环境

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-11-18 17:19:19-17:22:19（Asia/Shanghai） |
| 平台 | MetaX C500，64 GiB |
| 设备 | 2 x MetaX C500/C550，物理 GPU 3、4 |
| Driver / runtime | MetaX driver 3.3.12 / MACA 3.2.1.10 |
| InfiniLM build | 当前源码树；未记录独立 build ID |
| 模型 | `/data-aisoft/mechdancer/models/Qwen3-8B` |
| 模式 | `prefill-direct` |
| Attention backend | `flash-attn` |
| Graph request | enabled |
| Expected execution mode | `eager` |
| Cache | paged KV cache，block size 256，num blocks 64 |
| Max batched tokens | 18,432 |
| Tensor parallel size | 2 |
| Input length | 2,048 tokens / request |
| Batch size | 1..8 |
| Warmup | 每个形状 3 次 repeat |
| Measured repeats | 每个形状 30 次 repeat |
| Batch 执行顺序 | random seed 2026 |
| 干扰控制 | 物理专用 GPU 3、4；每 10 秒采样 `mx-smi` |

本报告对应的原始 JSONL 和 log 为本机运行产物，不随仓库提交：

```text
../../../benchmark-prefill-direct-metax-c500-tp2-qwen3-8b-input2048-batch1-8-clean-20261118.jsonl
../../../benchmark-prefill-direct-metax-c500-tp2-qwen3-8b-input2048-batch1-8-clean-20261118.log
```

## 构建与测试命令

C++ 扩展使用当前源码构建并安装：

```bash
xmake build _infinilm && xmake install _infinilm
```

性能测试命令：

```bash
BENCHMARK_GPU=3,4 \
INFINILM_MAX_NUM_BATCHED_TOKENS=18432 \
BENCHMARK_MODE=prefill-direct \
BATCH_SIZES="$(seq -s, 1 8)" \
INPUT_LENS=2048 \
BENCHMARK_REPEATS=30 \
BENCHMARK_OUTPUT=benchmark-prefill-direct-metax-c500-tp2-qwen3-8b-input2048-batch1-8-clean-20261118.jsonl \
./run_bench-metax.sh
```


脚本说明：上述命令记录的是 2026-11-18 运行时的调用方式；当时本地 `run_bench-metax.sh` 硬编码 `--tp 2`。该脚本被仓库 `.gitignore` 忽略，不随本报告提交；如果后续脚本被改为其他 TP 值，复现前必须先确认或改回 `--tp 2`，并保持 `BENCHMARK_GPU=3,4`。

## 执行模式与 Graph 覆盖

`run_bench-metax.sh` 请求开启 graph compiling，但当前 paged graph compiler 的 graph key 只匹配 decode-only 形状。packed prefill 形状不匹配该 key，因此本轮 expected execution mode 为 `eager`。
`expected_execution_mode` 是 benchmark 根据 compiler 规则推断的字段，不是 C++ runtime 回传的实际执行模式。

## 资源开销

| 项目 | 值 |
| --- | ---: |
| Workspace allocation | 每 rank 1,116.0 MiB，TP=2 合计 2,232.0 MiB |
| 权重加载时间 | 约 8.88 秒 |
| Paged cache blocks | 64 |
| 正式 repeat 数 | 240 |
| 本轮总耗时 | 约 180.7 秒 |
| GPU 3/4 温度范围 | 30C-53C |
| GPU 3/4 功耗范围 | 95W-300W |
| GPU 3/4 进程显存峰值 | 36,106 MiB |

## 测试口径

- 每个 sample 是一次物理 prefill forward。
- `token_count = batch_size * 2048`。
- `throughput = batch_size * 2048 / avg_forward_duration_s`。
- 计时边界为 `engine_forward_including_sampling`，包含 Python wrapper、C++ engine forward、RankWorker forward、TP 通信、model forward 和 sampling。
- 不包含 HTTP、JSON parsing、tokenization、scheduler queueing、`build_model_inputs`、detokenization 和 Python `generate()` 循环。
- 输出 token 不用于正确性判断。
- 分位数使用线性插值；每个 batch 30 个 repeat，p99 只能作为尾部参考。

## 核心结论

| 指标 | 结果 |
| --- | ---: |
| 最高 prefill 吞吐 | batch 8，14,450.36 tok/s |
| batch 1 平均 latency | 158.068 ms |
| batch 8 平均 latency | 1,133.813 ms |
| 最大 batch 8 吞吐 | 14,450.36 tok/s |

吞吐 Top 10 overall 结果：

| Batch | Avg ms | Throughput tok/s | Scaling |
| ---: | ---: | ---: | ---: |
| 8 | 1,133.813 | 14,450.36 | 1.115x |
| 7 | 997.290 | 14,374.96 | 1.109x |
| 4 | 571.671 | 14,329.91 | 1.106x |
| 6 | 861.327 | 14,266.36 | 1.101x |
| 5 | 724.650 | 14,130.95 | 1.091x |
| 3 | 438.783 | 14,002.36 | 1.081x |
| 2 | 302.769 | 13,528.45 | 1.044x |
| 1 | 158.068 | 12,956.48 | 1.000x |

## 趋势图

### 1. Batch size 与平均 latency

![average latency trend](assets/prefill-direct-metax-c500-tp2-qwen3-8b-input2048-batch1-8-eager-20261118-avg-latency-trend.png)

### 2. Batch size 与吞吐

![throughput trend](assets/prefill-direct-metax-c500-tp2-qwen3-8b-input2048-batch1-8-eager-20261118-throughput-trend.png)

### 3. Batch size 与吞吐 Scaling

![throughput scaling trend](assets/prefill-direct-metax-c500-tp2-qwen3-8b-input2048-batch1-8-eager-20261118-scaling-trend.png)


平均 latency 从 batch 1 的 158.068 ms 增至 batch 8 的 1,133.813 ms。吞吐峰值为 batch 8 的 14,450.36 tok/s。

## 关键批次与区间

| Batch / range | 观察 |
| --- | --- |
| batch 1 -> 8 | 平均 latency 从 158.068 ms 增至 1,133.813 ms，为 7.173x。 |
| batch 8 | 吞吐峰值，14,450.36 tok/s。 |
| batch 8 | per-request prefill latency 最低，为 141.73 ms/request。 |

## 数据稳定性

- 8 个 overall summary的 std 中位数为 0.634 ms。
- 最大 std 为 1.572 ms，出现在 batch 6。
- 最大变异系数约 0.400%，中位变异系数约 0.132%。
- P50、P95、P99 和 max 普遍接近。
- 每个 batch 只有 30 个 repeat 样本；p99 可用于观察尾部，正式尾部对比建议提高到 100+ repeat。

## 完整结果表

| Batch | Mode | Input tokens | Avg ms | Min ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Throughput tok/s | Scaling |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | eager | 2,048 | 158.068 | 156.769 | 157.993 | 159.097 | 159.474 | 159.581 | 0.632 | 12,956.48 | 1.000x |
| 2 | eager | 4,096 | 302.769 | 301.829 | 302.742 | 303.575 | 303.771 | 303.781 | 0.416 | 13,528.45 | 1.044x |
| 3 | eager | 6,144 | 438.783 | 437.483 | 438.804 | 439.640 | 439.769 | 439.809 | 0.560 | 14,002.36 | 1.081x |
| 4 | eager | 8,192 | 571.671 | 568.920 | 571.976 | 573.768 | 574.708 | 574.910 | 1.570 | 14,329.91 | 1.106x |
| 5 | eager | 10,240 | 724.650 | 722.912 | 724.824 | 725.376 | 725.644 | 725.731 | 0.648 | 14,130.95 | 1.091x |
| 6 | eager | 12,288 | 861.327 | 859.303 | 860.527 | 864.427 | 864.645 | 864.678 | 1.572 | 14,266.36 | 1.101x |
| 7 | eager | 14,336 | 997.290 | 995.697 | 997.227 | 998.342 | 998.780 | 998.940 | 0.637 | 14,374.96 | 1.109x |
| 8 | eager | 16,384 | 1,133.813 | 1,132.833 | 1,133.770 | 1,134.778 | 1,135.286 | 1,135.459 | 0.610 | 14,450.36 | 1.115x |
## 结果分析

1. 吞吐最高出现在 batch 8，为 14,450.36 tok/s。
2. 在本轮 batch `1..8` 范围内，吞吐在最大 batch 仍未回落；若资源允许，可继续扩大 batch 寻找饱和点。
3. 对比 MetaX TP=1 峰值 9,135.67 tok/s，本报告 TP=2 峰值为其 1.582x；占用两张卡后的每卡效率为 TP=1 的 0.791x。
4. 对比 NVIDIA A100 TP=2 峰值 22,020.10 tok/s，本报告峰值约为其 0.656x。
5. 正式 p99 对比建议将 `BENCHMARK_REPEATS` 提高到 100 或以上。

## 结论与建议

1. 在本轮 input length 2048、TP=2、batch `1..8` 范围内，**batch 8 吞吐最高**，约 14,450.36 tok/s。
2. batch 8 平均 forward latency 为 1,133.813 ms。
3. TP=2 相对 TP=1 提高总吞吐，但每卡效率低于 TP=1；是否采用 TP=2 应结合 SLO、总吞吐、时延和部署密度评估。
4. 所有样本均在专用 GPU 3、4 上采集；监控期间目标卡没有外来进程或多进程干扰。

## 数据完整性

```text
config records: 1
sample records: 240
overall summary records: 8
scaling records: 8
samples per batch: 30
expected execution modes: all eager
```
