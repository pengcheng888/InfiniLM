# Hygon BW1000 TP=2 direct benchmark 报告索引

本目录收录 Hygon BW1000 TP=2 上的 `prefill-direct` 和 `decode-direct` 性能报告。
所有报告使用 Qwen3-8B、flash-attn、paged KV cache、`max_num_batched_tokens=18432`、
warmup 3 次、正式测量 30 次 repeat。

## 运行环境

| 项目 | 值 |
| ---: | ---: |
| 日期 | 2026-11-23（Asia/Shanghai） |
| 平台 | Hygon BW1000 |
| 设备 | 2 x Hygon BW1000 HCU，64 GB HBM |
| Driver / runtime | 6.3.30-V1.4.1a / DTK 26.04-rc4 |
| 模型 | `/data/shared/.infinifabric-models-20261027-fef304a05440/Qwen3-8B--b968826d9c46dd6066d109eabc6255188de91218` |
| Tensor parallel | 2 |
| Warmup / measured repeat | 3 / 30 |
| Decode step | 每 repeat 4 step |
| 干扰控制 | 每 10 秒采样 `hy-smi` |


## 报告索引

| Input length | 模式 | Batch range | Max workload | 执行模式 | 报告 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | `prefill-direct` | 1..8 | 4,096 | eager | [prefill-direct Input 512](prefill-direct-hygon-bw1000-tp2-qwen3-8b-input512-batch1-8-eager-20261123.md) |
| 512 | `decode-direct` | 1..256 | 1..256 requests | graph | [decode-direct Input 512](decode-direct-hygon-bw1000-tp2-qwen3-8b-input512-batch1-256-graph-20261123.md) |
| 1024 | `prefill-direct` | 1..8 | 8,192 | eager | [prefill-direct Input 1024](prefill-direct-hygon-bw1000-tp2-qwen3-8b-input1024-batch1-8-eager-20261123.md) |
| 1024 | `decode-direct` | 1..256 | 1..256 requests | graph | [decode-direct Input 1024](decode-direct-hygon-bw1000-tp2-qwen3-8b-input1024-batch1-256-graph-20261123.md) |
| 2048 | `prefill-direct` | 1..8 | 16,384 | eager | [prefill-direct Input 2048](prefill-direct-hygon-bw1000-tp2-qwen3-8b-input2048-batch1-8-eager-20261123.md) |
| 2048 | `decode-direct` | 1..128 | 1..128 requests | graph | [decode-direct Input 2048](decode-direct-hygon-bw1000-tp2-qwen3-8b-input2048-batch1-128-graph-20261123.md) |
| 4096 | `prefill-direct` | 1..4 | 16,384 | eager | [prefill-direct Input 4096](prefill-direct-hygon-bw1000-tp2-qwen3-8b-input4096-batch1-4-eager-20261123.md) |
| 4096 | `decode-direct` | 1..64 | 1..64 requests | graph | [decode-direct Input 4096](decode-direct-hygon-bw1000-tp2-qwen3-8b-input4096-batch1-64-graph-20261123.md) |


## 核心结果

| Input | 模式 | 最佳 batch | 最佳吞吐 tok/s | Scaling vs batch 1 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | prefill-direct | 5 | 16,571.70 | 1.554x |
| 512 | decode-direct | 256 | 5,334.23 | 69.137x |
| 1024 | prefill-direct | 5 | 16,788.22 | 1.218x |
| 1024 | decode-direct | 256 | 4,134.71 | 59.777x |
| 2048 | prefill-direct | 8 | 16,244.36 | 1.128x |
| 2048 | decode-direct | 120 | 2,341.81 | 41.432x |
| 4096 | prefill-direct | 4 | 14,736.60 | 1.072x |
| 4096 | decode-direct | 60 | 1,241.38 | 30.085x |


## TP 扩展结果

| Input | 模式 | TP=2 峰值 tok/s | TP / TP=1 | TP / TP=2 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | decode-direct | 5,334.23 | 1.679x | 1.000x |
| 512 | prefill-direct | 16,571.70 | 1.587x | 1.000x |
| 1024 | decode-direct | 4,134.71 | 1.894x | 1.000x |
| 1024 | prefill-direct | 16,788.22 | 1.638x | 1.000x |
| 2048 | decode-direct | 2,341.81 | 1.681x | 1.000x |
| 2048 | prefill-direct | 16,244.36 | 1.679x | 1.000x |
| 4096 | decode-direct | 1,241.38 | 1.639x | 1.000x |
| 4096 | prefill-direct | 14,736.60 | 1.705x | 1.000x |


## 与 NVIDIA A100 / MetaX C500 对比

下表比较各平台同 TP 配置、同 input/mode 的峰值吞吐。平台硬件、驱动、runtime 和部分历史配置不同，该表只作为量级参考，不是严格控制变量实验。

| Input | 模式 | Hygon BW1000 | NVIDIA A100 | Hygon / A100 | MetaX C500 | Hygon / MetaX |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | decode-direct | batch 256，5,334.23 | batch 256，8,969.22 | 0.595x | batch 255，6,218.21 | 0.858x |
| 512 | prefill-direct | batch 5，16,571.70 | batch 8，21,452.86 | 0.772x | batch 7，14,090.26 | 1.176x |
| 1024 | decode-direct | batch 256，4,134.71 | batch 255，7,552.53 | 0.547x | batch 256，5,257.30 | 0.786x |
| 1024 | prefill-direct | batch 5，16,788.22 | batch 8，21,804.24 | 0.770x | batch 8，14,646.77 | 1.146x |
| 2048 | decode-direct | batch 120，2,341.81 | batch 124，4,873.34 | 0.481x | batch 128，3,751.00 | 0.624x |
| 2048 | prefill-direct | batch 8，16,244.36 | batch 8，22,020.10 | 0.738x | batch 8，14,450.36 | 1.124x |
| 4096 | decode-direct | batch 60，1,241.38 | batch 54，2,731.98 | 0.454x | batch 64，2,163.46 | 0.574x |
| 4096 | prefill-direct | batch 4，14,736.60 | batch 4，21,294.20 | 0.692x | batch 4，13,884.66 | 1.061x |


Input 1024 的历史平台报告存在 `max_num_batched_tokens` 差异；Hygon 本轮统一使用 18432。Hygon TP=1 input 1024 decode 有效范围是 1..192，与 MetaX TP=1 相同，小于 NVIDIA TP=1 报告的 1..256。

## 原始产物

Raw JSONL、log 和 `hy-smi` monitor 文件是本机运行产物，不随仓库提交。报告中保留仓库根目录的相对文件名用于本机追溯。
