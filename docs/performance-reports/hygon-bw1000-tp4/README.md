# Hygon BW1000 TP=4 direct benchmark 报告索引

本目录收录 Hygon BW1000 TP=4 上的 `prefill-direct` 和 `decode-direct` 性能报告。
所有报告使用 Qwen3-8B、flash-attn、paged KV cache、`max_num_batched_tokens=18432`、
warmup 3 次、正式测量 30 次 repeat。

## 运行环境

| 项目 | 值 |
| ---: | ---: |
| 日期 | 2026-11-23（Asia/Shanghai） |
| 平台 | Hygon BW1000 |
| 设备 | 4 x Hygon BW1000 HCU，64 GB HBM |
| Driver / runtime | 6.3.30-V1.4.1a / DTK 26.04-rc4 |
| 模型 | `/data/shared/.infinifabric-models-20261027-fef304a05440/Qwen3-8B--b968826d9c46dd6066d109eabc6255188de91218` |
| Tensor parallel | 4 |
| Warmup / measured repeat | 3 / 30 |
| Decode step | 每 repeat 4 step |
| 干扰控制 | 每 10 秒采样 `hy-smi` |


## 报告索引

| Input length | 模式 | Batch range | Max workload | 执行模式 | 报告 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | `prefill-direct` | 1..8 | 4,096 | eager | [prefill-direct Input 512](prefill-direct-hygon-bw1000-tp4-qwen3-8b-input512-batch1-8-eager-20261123.md) |
| 512 | `decode-direct` | 1..256 | 1..256 requests | graph | [decode-direct Input 512](decode-direct-hygon-bw1000-tp4-qwen3-8b-input512-batch1-256-graph-20261123.md) |
| 1024 | `prefill-direct` | 1..8 | 8,192 | eager | [prefill-direct Input 1024](prefill-direct-hygon-bw1000-tp4-qwen3-8b-input1024-batch1-8-eager-20261123.md) |
| 1024 | `decode-direct` | 1..256 | 1..256 requests | graph | [decode-direct Input 1024](decode-direct-hygon-bw1000-tp4-qwen3-8b-input1024-batch1-256-graph-20261123.md) |
| 2048 | `prefill-direct` | 1..8 | 16,384 | eager | [prefill-direct Input 2048](prefill-direct-hygon-bw1000-tp4-qwen3-8b-input2048-batch1-8-eager-20261123.md) |
| 2048 | `decode-direct` | 1..128 | 1..128 requests | graph | [decode-direct Input 2048](decode-direct-hygon-bw1000-tp4-qwen3-8b-input2048-batch1-128-graph-20261123.md) |
| 4096 | `prefill-direct` | 1..4 | 16,384 | eager | [prefill-direct Input 4096](prefill-direct-hygon-bw1000-tp4-qwen3-8b-input4096-batch1-4-eager-20261123.md) |
| 4096 | `decode-direct` | 1..64 | 1..64 requests | graph | [decode-direct Input 4096](decode-direct-hygon-bw1000-tp4-qwen3-8b-input4096-batch1-64-graph-20261123.md) |


## 核心结果

| Input | 模式 | 最佳 batch | 最佳吞吐 tok/s | Scaling vs batch 1 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | prefill-direct | 5 | 24,751.22 | 1.697x |
| 512 | decode-direct | 256 | 7,169.85 | 80.549x |
| 1024 | prefill-direct | 8 | 25,516.22 | 1.380x |
| 1024 | decode-direct | 240 | 5,899.98 | 75.759x |
| 2048 | prefill-direct | 8 | 25,208.09 | 1.192x |
| 2048 | decode-direct | 120 | 3,467.80 | 55.775x |
| 4096 | prefill-direct | 4 | 23,285.54 | 1.106x |
| 4096 | decode-direct | 63 | 2,021.80 | 45.130x |


## TP 扩展结果

| Input | 模式 | TP=4 峰值 tok/s | TP / TP=1 | TP / TP=2 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | decode-direct | 7,169.85 | 2.257x | 1.344x |
| 512 | prefill-direct | 24,751.22 | 2.370x | 1.494x |
| 1024 | decode-direct | 5,899.98 | 2.702x | 1.427x |
| 1024 | prefill-direct | 25,516.22 | 2.489x | 1.520x |
| 2048 | decode-direct | 3,467.80 | 2.489x | 1.481x |
| 2048 | prefill-direct | 25,208.09 | 2.606x | 1.552x |
| 4096 | decode-direct | 2,021.80 | 2.670x | 1.629x |
| 4096 | prefill-direct | 23,285.54 | 2.695x | 1.580x |


## 与 NVIDIA A100 / MetaX C500 对比

下表比较各平台同 TP 配置、同 input/mode 的峰值吞吐。平台硬件、驱动、runtime 和部分历史配置不同，该表只作为量级参考，不是严格控制变量实验。

| Input | 模式 | Hygon BW1000 | NVIDIA A100 | Hygon / A100 | MetaX C500 | Hygon / MetaX |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | decode-direct | batch 256，7,169.85 | batch 256，12,317.88 | 0.582x | batch 255，9,417.38 | 0.761x |
| 512 | prefill-direct | batch 5，24,751.22 | batch 8，34,781.56 | 0.712x | batch 8，24,617.17 | 1.005x |
| 1024 | decode-direct | batch 240，5,899.98 | batch 256，10,683.02 | 0.552x | batch 255，8,366.76 | 0.705x |
| 1024 | prefill-direct | batch 8，25,516.22 | batch 8，35,960.64 | 0.710x | batch 8，25,917.55 | 0.985x |
| 2048 | decode-direct | batch 120，3,467.80 | batch 108，7,058.49 | 0.491x | batch 128，5,665.03 | 0.612x |
| 2048 | prefill-direct | batch 8，25,208.09 | batch 8，36,634.03 | 0.688x | batch 8，25,896.44 | 0.973x |
| 4096 | decode-direct | batch 63，2,021.80 | batch 64，4,181.42 | 0.484x | batch 63，3,304.40 | 0.612x |
| 4096 | prefill-direct | batch 4，23,285.54 | batch 4，35,606.70 | 0.654x | batch 4，25,041.42 | 0.930x |


Input 1024 的历史平台报告存在 `max_num_batched_tokens` 差异；Hygon 本轮统一使用 18432。Hygon TP=1 input 1024 decode 有效范围是 1..192，与 MetaX TP=1 相同，小于 NVIDIA TP=1 报告的 1..256。

## 原始产物

Raw JSONL、log 和 `hy-smi` monitor 文件是本机运行产物，不随仓库提交。报告中保留仓库根目录的相对文件名用于本机追溯。
