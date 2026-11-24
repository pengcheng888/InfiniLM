# Hygon BW1000 TP=1 direct benchmark 报告索引

本目录收录 Hygon BW1000 TP=1 上的 `prefill-direct` 和 `decode-direct` 性能报告。
所有报告使用 Qwen3-8B、flash-attn、paged KV cache、`max_num_batched_tokens=18432`、
warmup 3 次、正式测量 30 次 repeat。

## 运行环境

| 项目 | 值 |
| ---: | ---: |
| 日期 | 2026-11-23（Asia/Shanghai） |
| 平台 | Hygon BW1000 |
| 设备 | 1 x Hygon BW1000 HCU，64 GB HBM |
| Driver / runtime | 6.3.30-V1.4.1a / DTK 26.04-rc4 |
| 模型 | `/data/shared/.infinifabric-models-20261027-fef304a05440/Qwen3-8B--b968826d9c46dd6066d109eabc6255188de91218` |
| Tensor parallel | 1 |
| Warmup / measured repeat | 3 / 30 |
| Decode step | 每 repeat 4 step |
| 干扰控制 | 每 10 秒采样 `hy-smi` |


## 报告索引

| Input length | 模式 | Batch range | Max workload | 执行模式 | 报告 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | `prefill-direct` | 1..8 | 4,096 | eager | [prefill-direct Input 512](prefill-direct-hygon-bw1000-qwen3-8b-input512-batch1-8-eager-20261123.md) |
| 512 | `decode-direct` | 1..256 | 1..256 requests | graph | [decode-direct Input 512](decode-direct-hygon-bw1000-qwen3-8b-input512-batch1-256-graph-20261123.md) |
| 1024 | `prefill-direct` | 1..8 | 8,192 | eager | [prefill-direct Input 1024](prefill-direct-hygon-bw1000-qwen3-8b-input1024-batch1-8-eager-20261123.md) |
| 1024 | `decode-direct` | 1..192 | 1..192 requests | graph | [decode-direct Input 1024](decode-direct-hygon-bw1000-qwen3-8b-input1024-batch1-192-graph-20261123.md) |
| 2048 | `prefill-direct` | 1..8 | 16,384 | eager | [prefill-direct Input 2048](prefill-direct-hygon-bw1000-qwen3-8b-input2048-batch1-8-eager-20261123.md) |
| 2048 | `decode-direct` | 1..128 | 1..128 requests | graph | [decode-direct Input 2048](decode-direct-hygon-bw1000-qwen3-8b-input2048-batch1-128-graph-20261123.md) |
| 4096 | `prefill-direct` | 1..4 | 16,384 | eager | [prefill-direct Input 4096](prefill-direct-hygon-bw1000-qwen3-8b-input4096-batch1-4-eager-20261123.md) |
| 4096 | `decode-direct` | 1..64 | 1..64 requests | graph | [decode-direct Input 4096](decode-direct-hygon-bw1000-qwen3-8b-input4096-batch1-64-graph-20261123.md) |


## 核心结果

| Input | 模式 | 最佳 batch | 最佳吞吐 tok/s | Scaling vs batch 1 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | prefill-direct | 5 | 10,441.88 | 1.551x |
| 512 | decode-direct | 254 | 3,176.67 | 71.647x |
| 1024 | prefill-direct | 5 | 10,252.30 | 1.118x |
| 1024 | decode-direct | 180 | 2,183.33 | 53.072x |
| 2048 | prefill-direct | 8 | 9,674.84 | 1.071x |
| 2048 | decode-direct | 128 | 1,393.35 | 38.121x |
| 4096 | prefill-direct | 4 | 8,640.70 | 1.043x |
| 4096 | decode-direct | 60 | 757.36 | 25.640x |


## 与 NVIDIA A100 / MetaX C500 对比

下表比较各平台同 TP 配置、同 input/mode 的峰值吞吐。平台硬件、驱动、runtime 和部分历史配置不同，该表只作为量级参考，不是严格控制变量实验。

| Input | 模式 | Hygon BW1000 | NVIDIA A100 | Hygon / A100 | MetaX C500 | Hygon / MetaX |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | decode-direct | batch 254，3,176.67 | batch 256，6,409.31 | 0.496x | batch 256，4,552.35 | 0.698x |
| 512 | prefill-direct | batch 5，10,441.88 | batch 5，12,489.58 | 0.836x | batch 7，9,310.41 | 1.122x |
| 1024 | decode-direct | batch 180，2,183.33 | batch 256，5,044.65 | 0.433x | batch 192，3,418.89 | 0.639x |
| 1024 | prefill-direct | batch 5，10,252.30 | batch 5，12,514.29 | 0.819x | batch 7，9,298.77 | 1.103x |
| 2048 | decode-direct | batch 128，1,393.35 | batch 128，3,240.36 | 0.430x | batch 104，2,394.80 | 0.582x |
| 2048 | prefill-direct | batch 8，9,674.84 | batch 7，12,322.39 | 0.785x | batch 8，9,135.67 | 1.059x |
| 4096 | decode-direct | batch 60，757.36 | batch 64，1,713.84 | 0.442x | batch 64，1,339.42 | 0.565x |
| 4096 | prefill-direct | batch 4，8,640.70 | batch 4，11,842.50 | 0.730x | batch 4，8,681.30 | 0.995x |


Input 1024 的历史平台报告存在 `max_num_batched_tokens` 差异；Hygon 本轮统一使用 18432。Hygon TP=1 input 1024 decode 有效范围是 1..192，与 MetaX TP=1 相同，小于 NVIDIA TP=1 报告的 1..256。

## 原始产物

Raw JSONL、log 和 `hy-smi` monitor 文件是本机运行产物，不随仓库提交。报告中保留仓库根目录的相对文件名用于本机追溯。
