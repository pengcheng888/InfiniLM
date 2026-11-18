# NVIDIA A100 TP=2 direct benchmark 报告索引

本目录收录 NVIDIA A100-SXM4-80GB、TP=2 上的 `prefill-direct` 和
`decode-direct` 性能报告。所有报告使用 Qwen3-8B、flash-attn、paged KV cache、
`max_num_batched_tokens=18432`、warmup 3 次、正式测量 30 次 repeat。

## 报告索引

| Input length | 模式 | Batch range | 执行模式 | 报告 |
| ---: | --- | --- | --- | --- |
| 512 | `prefill-direct` | 1..8 | eager | [Prefill-Direct Input 512 TP=2](prefill-direct-nvidia-a100-tp2-qwen3-8b-input512-batch1-8-eager-20261118.md) |
| 512 | `decode-direct` | 1..256 | graph | [Decode-Direct Input 512 TP=2](decode-direct-nvidia-a100-tp2-qwen3-8b-input512-batch1-256-graph-20261118.md) |
| 1024 | `prefill-direct` | 1..8 | eager | [Prefill-Direct Input 1024 TP=2](prefill-direct-nvidia-a100-tp2-qwen3-8b-input1024-batch1-8-eager-20261118.md) |
| 1024 | `decode-direct` | 1..256 | graph | [Decode-Direct Input 1024 TP=2](decode-direct-nvidia-a100-tp2-qwen3-8b-input1024-batch1-256-graph-20261118.md) |
| 2048 | `prefill-direct` | 1..8 | eager | [Prefill-Direct Input 2048 TP=2](prefill-direct-nvidia-a100-tp2-qwen3-8b-input2048-batch1-8-eager-20261118.md) |
| 2048 | `decode-direct` | 1..128 | graph | [Decode-Direct Input 2048 TP=2](decode-direct-nvidia-a100-tp2-qwen3-8b-input2048-batch1-128-graph-20261118.md) |
| 4096 | `prefill-direct` | 1..4 | eager | [Prefill-Direct Input 4096 TP=2](prefill-direct-nvidia-a100-tp2-qwen3-8b-input4096-batch1-4-eager-20261118.md) |
| 4096 | `decode-direct` | 1..64 | graph | [Decode-Direct Input 4096 TP=2](decode-direct-nvidia-a100-tp2-qwen3-8b-input4096-batch1-64-graph-20261118.md) |

## 峰值结果与 TP=1 对比

下表比较本目录 TP=2 与 `nvidia-a100-tp1/` 目录 TP=1 报告的各自峰值吞吐。
两组报告均使用 A100-SXM4-80GB、Qwen3-8B 和 flash-attn。

| Input length | 模式 | TP=1 峰值 | TP=2 峰值 | TP=2 / TP=1 吞吐 |
| ---: | --- | ---: | ---: | ---: |
| 512 | prefill | 12,489.58 tok/s | batch 8，21,452.86 tok/s | 1.718x |
| 512 | decode | 6,409.31 tok/s | batch 256，8,969.22 tok/s | 1.399x |
| 1024 | prefill | 12,514.29 tok/s | batch 8，21,804.24 tok/s | 1.742x |
| 1024 | decode | 5,044.65 tok/s | batch 255，7,552.53 tok/s | 1.497x |
| 2048 | prefill | 12,322.39 tok/s | batch 8，22,020.10 tok/s | 1.787x |
| 2048 | decode | 3,240.36 tok/s | batch 124，4,873.34 tok/s | 1.504x |
| 4096 | prefill | 11,842.50 tok/s | batch 4，21,294.20 tok/s | 1.798x |
| 4096 | decode | 1,713.84 tok/s | batch 54，2,731.98 tok/s | 1.594x |

TP=2 在本轮 direct benchmark 中显著提高峰值吞吐：prefill 加速为
1.718x 到 1.798x，decode 加速为 1.399x 到 1.594x，并且各组合峰值处的
平均 latency 均低于 TP=1。代价是占用两张 A100 并引入 TP 通信；实际选型
应同时评估目标 SLO、总吞吐和每卡吞吐。

Input 1024 的 TP=1 参考报告使用 `max_num_batched_tokens=9216`，本目录
TP=2 使用 18432；因此 input 1024 的跨 TP 对比不是严格控制变量结果。
input 512、2048 和 4096 的参考报告均使用 18432。

## 原始产物

Raw JSONL 和 log 文件是本机运行产物，不随仓库提交。各报告中保留了指向仓库根目录原始产物的相对路径，仅用于本机追溯。
