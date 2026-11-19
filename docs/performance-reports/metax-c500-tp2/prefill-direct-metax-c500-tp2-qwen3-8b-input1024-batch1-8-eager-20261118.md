# Prefill-Direct Input 1024 TP=2 性能报告

## 运行环境

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-11-18 17:17:29-17:19:03（Asia/Shanghai） |
| 平台 | MetaX C500，64 GiB |
| 设备 | 2 x MetaX C500/C550，物理 GPU 3、4 |
| Driver / runtime | MetaX driver 3.3.12 / MACA 3.2.1.10 |
| InfiniLM build | 当前源码树；未记录独立 build ID |
| 模型 | `/data-aisoft/mechdancer/models/Qwen3-8B` |
| 模式 | `prefill-direct` |
| Attention backend | `flash-attn` |
| Graph request | enabled |
| Expected execution mode | `eager` |
| Cache | paged KV cache，block size 256，num blocks 32 |
| Max batched tokens | 9,216 |
| Tensor parallel size | 2 |
| Input length | 1,024 tokens / request |
| Batch size | 1..8 |
| Warmup | 每个形状 3 次 repeat |
| Measured repeats | 每个形状 30 次 repeat |
| Batch 执行顺序 | random seed 2026 |
| 干扰控制 | 物理专用 GPU 3、4；每 10 秒采样 `mx-smi` |

本报告对应的原始 JSONL 和 log 为本机运行产物，不随仓库提交：

```text
../../../benchmark-prefill-direct-metax-c500-tp2-qwen3-8b-input1024-batch1-8-clean-20261118.jsonl
../../../benchmark-prefill-direct-metax-c500-tp2-qwen3-8b-input1024-batch1-8-clean-20261118.log
```

## 构建与测试命令

C++ 扩展使用当前源码构建并安装：

```bash
xmake build _infinilm && xmake install _infinilm
```

性能测试命令：

```bash
BENCHMARK_GPU=3,4 \
INFINILM_MAX_NUM_BATCHED_TOKENS=9216 \
BENCHMARK_MODE=prefill-direct \
BATCH_SIZES="$(seq -s, 1 8)" \
INPUT_LENS=1024 \
BENCHMARK_REPEATS=30 \
BENCHMARK_OUTPUT=benchmark-prefill-direct-metax-c500-tp2-qwen3-8b-input1024-batch1-8-clean-20261118.jsonl \
./run_bench-metax.sh
```


脚本说明：上述命令记录的是 2026-11-18 运行时的调用方式；当时本地 `run_bench-metax.sh` 硬编码 `--tp 2`。该脚本被仓库 `.gitignore` 忽略，不随本报告提交；如果后续脚本被改为其他 TP 值，复现前必须先确认或改回 `--tp 2`，并保持 `BENCHMARK_GPU=3,4`。

## 执行模式与 Graph 覆盖

`run_bench-metax.sh` 请求开启 graph compiling，但当前 paged graph compiler 的 graph key 只匹配 decode-only 形状。packed prefill 形状不匹配该 key，因此本轮 expected execution mode 为 `eager`。
`expected_execution_mode` 是 benchmark 根据 compiler 规则推断的字段，不是 C++ runtime 回传的实际执行模式。

## 资源开销

| 项目 | 值 |
| --- | ---: |
| Workspace allocation | 每 rank 558.0 MiB，TP=2 合计 1,116.0 MiB |
| 权重加载时间 | 约 8.55 秒 |
| Paged cache blocks | 32 |
| 正式 repeat 数 | 240 |
| 本轮总耗时 | 约 94.9 秒 |
| GPU 3/4 温度范围 | 30C-53C |
| GPU 3/4 功耗范围 | 95W-300W |
| GPU 3/4 进程显存峰值 | 36,106 MiB |

## 测试口径

- 每个 sample 是一次物理 prefill forward。
- `token_count = batch_size * 1024`。
- `throughput = batch_size * 1024 / avg_forward_duration_s`。
- 计时边界为 `engine_forward_including_sampling`，包含 Python wrapper、C++ engine forward、RankWorker forward、TP 通信、model forward 和 sampling。
- 不包含 HTTP、JSON parsing、tokenization、scheduler queueing、`build_model_inputs`、detokenization 和 Python `generate()` 循环。
- 输出 token 不用于正确性判断。
- 分位数使用线性插值；每个 batch 30 个 repeat，p99 只能作为尾部参考。

## 核心结论

| 指标 | 结果 |
| --- | ---: |
| 最高 prefill 吞吐 | batch 8，14,646.77 tok/s |
| batch 1 平均 latency | 80.885 ms |
| batch 8 平均 latency | 559.304 ms |
| 最大 batch 8 吞吐 | 14,646.77 tok/s |

吞吐 Top 10 overall 结果：

| Batch | Avg ms | Throughput tok/s | Scaling |
| ---: | ---: | ---: | ---: |
| 8 | 559.304 | 14,646.77 | 1.157x |
| 7 | 496.069 | 14,449.61 | 1.141x |
| 6 | 429.464 | 14,306.19 | 1.130x |
| 3 | 221.457 | 13,871.76 | 1.096x |
| 4 | 296.277 | 13,824.90 | 1.092x |
| 5 | 372.809 | 13,733.57 | 1.085x |
| 2 | 152.328 | 13,444.69 | 1.062x |
| 1 | 80.885 | 12,659.90 | 1.000x |

## 趋势图

### 1. Batch size 与平均 latency

![average latency trend](assets/prefill-direct-metax-c500-tp2-qwen3-8b-input1024-batch1-8-eager-20261118-avg-latency-trend.png)

### 2. Batch size 与吞吐

![throughput trend](assets/prefill-direct-metax-c500-tp2-qwen3-8b-input1024-batch1-8-eager-20261118-throughput-trend.png)

### 3. Batch size 与吞吐 Scaling

![throughput scaling trend](assets/prefill-direct-metax-c500-tp2-qwen3-8b-input1024-batch1-8-eager-20261118-scaling-trend.png)


平均 latency 从 batch 1 的 80.885 ms 增至 batch 8 的 559.304 ms。吞吐峰值为 batch 8 的 14,646.77 tok/s。

## 关键批次与区间

| Batch / range | 观察 |
| --- | --- |
| batch 1 -> 8 | 平均 latency 从 80.885 ms 增至 559.304 ms，为 6.915x。 |
| batch 8 | 吞吐峰值，14,646.77 tok/s。 |
| batch 8 | per-request prefill latency 最低，为 69.91 ms/request。 |

## 数据稳定性

- 8 个 overall summary的 std 中位数为 0.407 ms。
- 最大 std 为 1.542 ms，出现在 batch 8。
- 最大变异系数约 1.054%，中位变异系数约 0.172%。
- P50、P95、P99 和 max 普遍接近。
- 每个 batch 只有 30 个 repeat 样本；p99 可用于观察尾部，正式尾部对比建议提高到 100+ repeat。

## 完整结果表

| Batch | Mode | Input tokens | Avg ms | Min ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Throughput tok/s | Scaling |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | eager | 1,024 | 80.885 | 79.251 | 81.236 | 81.833 | 82.263 | 82.346 | 0.853 | 12,659.90 | 1.000x |
| 2 | eager | 2,048 | 152.328 | 151.704 | 152.411 | 152.931 | 153.057 | 153.087 | 0.358 | 13,444.69 | 1.062x |
| 3 | eager | 3,072 | 221.457 | 220.661 | 221.486 | 222.188 | 222.224 | 222.234 | 0.401 | 13,871.76 | 1.096x |
| 4 | eager | 4,096 | 296.277 | 295.240 | 296.278 | 296.771 | 296.842 | 296.853 | 0.326 | 13,824.90 | 1.092x |
| 5 | eager | 5,120 | 372.809 | 371.799 | 372.829 | 373.288 | 373.485 | 373.564 | 0.398 | 13,733.57 | 1.085x |
| 6 | eager | 6,144 | 429.464 | 428.040 | 429.594 | 430.403 | 430.617 | 430.679 | 0.703 | 14,306.19 | 1.130x |
| 7 | eager | 7,168 | 496.069 | 495.495 | 496.026 | 496.521 | 497.445 | 497.777 | 0.413 | 14,449.61 | 1.141x |
| 8 | eager | 8,192 | 559.304 | 557.964 | 558.774 | 562.822 | 563.835 | 564.164 | 1.542 | 14,646.77 | 1.157x |
## 结果分析

1. 吞吐最高出现在 batch 8，为 14,646.77 tok/s。
2. 在本轮 batch `1..8` 范围内，吞吐在最大 batch 仍未回落；若资源允许，可继续扩大 batch 寻找饱和点。
3. 对比 MetaX TP=1 峰值 9,298.77 tok/s，本报告 TP=2 峰值为其 1.575x；占用两张卡后的每卡效率为 TP=1 的 0.788x。
4. 对比 NVIDIA A100 TP=2 峰值 21,804.24 tok/s，本报告峰值约为其 0.672x。
5. 正式 p99 对比建议将 `BENCHMARK_REPEATS` 提高到 100 或以上。

## 结论与建议

1. 在本轮 input length 1024、TP=2、batch `1..8` 范围内，**batch 8 吞吐最高**，约 14,646.77 tok/s。
2. batch 8 平均 forward latency 为 559.304 ms。
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
