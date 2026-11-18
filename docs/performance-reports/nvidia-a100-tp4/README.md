# NVIDIA A100 TP=4 direct benchmark 报告索引

本目录收录 NVIDIA A100-SXM4-80GB、TP=4 上的 `prefill-direct` 和 `decode-direct` 性能报告。所有报告使用 Qwen3-8B、flash-attn、paged KV cache、`max_num_batched_tokens=18432`、warmup 3 次、正式测量 30 次 repeat。

## 报告索引

| Input length | 模式 | Batch range | 执行模式 | 报告 |
| ---: | --- | --- | --- | --- |
| 512 | `prefill-direct` | 1..8 | eager | [Prefill-Direct Input 512 TP=4](prefill-direct-nvidia-a100-tp4-qwen3-8b-input512-batch1-8-eager-20261118.md) |
| 512 | `decode-direct` | 1..256 | graph | [Decode-Direct Input 512 TP=4](decode-direct-nvidia-a100-tp4-qwen3-8b-input512-batch1-256-graph-20261118.md) |
| 1024 | `prefill-direct` | 1..8 | eager | [Prefill-Direct Input 1024 TP=4](prefill-direct-nvidia-a100-tp4-qwen3-8b-input1024-batch1-8-eager-20261118.md) |
| 1024 | `decode-direct` | 1..256 | graph | [Decode-Direct Input 1024 TP=4](decode-direct-nvidia-a100-tp4-qwen3-8b-input1024-batch1-256-graph-20261118.md) |
| 2048 | `prefill-direct` | 1..8 | eager | [Prefill-Direct Input 2048 TP=4](prefill-direct-nvidia-a100-tp4-qwen3-8b-input2048-batch1-8-eager-20261118.md) |
| 2048 | `decode-direct` | 1..128 | graph | [Decode-Direct Input 2048 TP=4](decode-direct-nvidia-a100-tp4-qwen3-8b-input2048-batch1-128-graph-20261118.md) |
| 4096 | `prefill-direct` | 1..4 | eager | [Prefill-Direct Input 4096 TP=4](prefill-direct-nvidia-a100-tp4-qwen3-8b-input4096-batch1-4-eager-20261118.md) |
| 4096 | `decode-direct` | 1..64 | graph | [Decode-Direct Input 4096 TP=4](decode-direct-nvidia-a100-tp4-qwen3-8b-input4096-batch1-64-graph-20261118.md) |

## 峰值结果与 TP 对比

| Input length | 模式 | TP=1 峰值 | TP=2 峰值 | TP=4 峰值 | TP=4 / TP=2 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | prefill | 12,489.58 tok/s | 21,452.86 tok/s | batch 8，34,781.56 tok/s | 1.621x |
| 512 | decode | 6,409.31 tok/s | 8,969.22 tok/s | batch 256，12,317.88 tok/s | 1.373x |
| 1024 | prefill | 12,514.29 tok/s | 21,804.24 tok/s | batch 8，35,960.64 tok/s | 1.649x |
| 1024 | decode | 5,044.65 tok/s | 7,552.53 tok/s | batch 256，10,683.02 tok/s | 1.414x |
| 2048 | prefill | 12,322.39 tok/s | 22,020.10 tok/s | batch 8，36,634.03 tok/s | 1.664x |
| 2048 | decode | 3,240.36 tok/s | 4,873.34 tok/s | batch 108，7,058.49 tok/s | 1.448x |
| 4096 | prefill | 11,842.50 tok/s | 21,294.20 tok/s | batch 4，35,606.70 tok/s | 1.672x |
| 4096 | decode | 1,713.84 tok/s | 2,731.98 tok/s | batch 64，4,181.42 tok/s | 1.531x |

TP=4 相比 TP=2 的峰值吞吐继续提升：prefill 为 1.621x 到 1.672x，decode 为
1.373x 到 1.531x。但 TP=4 占用 4 张 A100 并引入更多通信；在本轮 8 个组合中，
TP=2 的每卡峰值吞吐均高于 TP=4，因此从每卡效率看 TP=2 更均衡。

TP=1 input 1024 参考报告使用 `max_num_batched_tokens=9216`，TP=2 和 TP=4 使用 18432；因此 input 1024 的 TP=1 对比不是严格控制变量结果。

## 原始产物

Raw JSONL 和 log 文件是本机运行产物，不随仓库提交。各报告保留了指向仓库根目录原始产物的相对路径，仅用于本机追溯。
