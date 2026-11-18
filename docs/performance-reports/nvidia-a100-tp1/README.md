# NVIDIA A100 direct benchmark 报告索引

本目录收录 NVIDIA A100-SXM4-80GB 上的 `prefill-direct` 和 `decode-direct`
性能报告。所有报告使用 Qwen3-8B、flash-attn 和 paged KV cache。

## 报告索引

| Input length | 模式 | Batch range | Max batched tokens | 执行模式 | 报告 |
| ---: | --- | --- | ---: | --- | --- |
| 512 | `prefill-direct` | 1..8 | 18432 | eager | [Prefill-Direct Input 512](prefill-direct-nvidia-a100-qwen3-8b-input512-batch1-8-eager-20261112.md) |
| 512 | `decode-direct` | 1..256 | 18432 | graph | [Decode-Direct Input 512 Graph 1..256](decode-direct-nvidia-a100-qwen3-8b-input512-batch1-256-graph-20261112.md) |
| 1024 | `prefill-direct` | 1..8 | 9216 | eager | [Prefill-Direct Input 1024](prefill-direct-nvidia-a100-qwen3-8b-input1024-batch1-8-eager-20261111.md) |
| 1024 | `decode-direct` | 1..256 | 9216 | graph | [Decode-Direct Input 1024 Graph 1..256](decode-direct-nvidia-a100-qwen3-8b-input1024-batch1-256-graph-20261111.md) |
| 2048 | `prefill-direct` | 1..8 | 18432 | eager | [Prefill-Direct Input 2048](prefill-direct-nvidia-a100-qwen3-8b-input2048-batch1-8-eager-20261111.md) |
| 2048 | `decode-direct` | 1..128 | 18432 | graph | [Decode-Direct Input 2048 Graph 1..128](decode-direct-nvidia-a100-qwen3-8b-input2048-batch1-128-graph-20261111.md) |
| 4096 | `prefill-direct` | 1..4 | 18432 | eager | [Prefill-Direct Input 4096](prefill-direct-nvidia-a100-qwen3-8b-input4096-batch1-4-eager-20261113.md) |
| 4096 | `decode-direct` | 1..64 | 18432 | graph | [Decode-Direct Input 4096 Graph 1..64](decode-direct-nvidia-a100-qwen3-8b-input4096-batch1-64-graph-20261113.md) |

## 配置差异说明

Input 1024 报告生成于默认 `max_num_batched_tokens=9216` 时；input 512、
input 2048 和 input 4096 报告生成于默认 `max_num_batched_tokens=18432` 时。
因此这些报告不能视为仅 input length 不同的严格控制变量实验。

如果需要跨 input length 的严格对比，应在当前默认 `18432` 下重新测试
input 1024。

Input 2048 的 decode-direct 曾尝试 batch `1..256` 全 graph 编译，但在
A100 80GB 上初始化阶段发生 CUDA malloc OOM；成功报告使用 batch `1..128`。
该尝试记录在 input 2048 decode 报告中。

Input 4096 测试按显存约束限制了 batch 范围：prefill-direct 使用 batch
`1..4`，其中 batch 4 的 16384 tokens 仍低于 `max_num_batched_tokens=18432`；
decode-direct 使用全 graph batch `1..64`。

## 原始产物

Raw JSONL 和 log 文件是本机运行产物，不随仓库提交。各报告中保留了指向
仓库根目录原始产物的相对路径，仅用于本机追溯。
