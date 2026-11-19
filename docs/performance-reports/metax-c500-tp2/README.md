# MetaX C500 TP=2 direct benchmark 报告索引

本目录收录 MetaX C500、TP=2 上的 `prefill-direct` 和 `decode-direct` 无干扰性能报告。全部报告使用 Qwen3-8B、flash-attn、paged KV cache、warmup 3 次、正式测量 30 次 repeat。Input 1024 使用 `max_num_batched_tokens=9216`，其余组合使用 18432。

## 测试环境

| 项目 | 值 |
| --- | --- |
| 平台 | MetaX C500，64 GiB |
| Driver / runtime | MetaX driver 3.3.12 / MACA 3.2.1.10 |
| 设备 | 物理专用 GPU 3、4 |
| 模型 | `/data-aisoft/mechdancer/models/Qwen3-8B` |
| Attention | `flash-attn` |
| Tensor parallel | 2 |
| Warmup / 正式 repeat | 每形状 3 次 / 30 次 |
| Decode step | 每 repeat 4 step |
| 干扰控制 | 每 10 秒采样 `mx-smi` |

## 干扰检查

监控文件：`benchmark-metax-c500-tp2-gpu-monitor-20261118.log`。

- 监控间隔：10 秒。
- 采样点：318 个。
- 时间范围：2026-11-18 17:15:49 - 2026-11-18 18:10:02。
- GPU 3/4 外来进程采样点：0。
- GPU 3/4 多进程采样点：0。
- GPU 3/4 进程显存峰值：36,106 MiB。
- GPU 3/4 温度范围：30C-53C。
- GPU 3/4 功耗范围：95W-300W。
- 测试结束后 GPU 3/4 无残留 benchmark 进程。

## 测试脚本说明

TP=2 报告产生时，本地 `run_bench-metax.sh` 硬编码 `--tp 2`，并使用物理 GPU 3、4。该脚本被 `.gitignore` 忽略，不随仓库提交；单项报告中的命令是历史调用记录。复现前需确认脚本已改回 `--tp 2`，避免把当前工作区中可能后续改为其他 TP 值的脚本误用于 TP=2 数据。

## 报告索引

| Input length | 模式 | Batch range | Max batched tokens | 执行模式 | 报告 |
| ---: | --- | --- | ---: | --- | --- |
| 512 | `prefill-direct` | 1..8 | 18,432 | eager | [prefill-direct Input 512 TP=2](prefill-direct-metax-c500-tp2-qwen3-8b-input512-batch1-8-eager-20261118.md) |
| 512 | `decode-direct` | 1..256 | 18,432 | graph | [decode-direct Input 512 TP=2](decode-direct-metax-c500-tp2-qwen3-8b-input512-batch1-256-graph-20261118.md) |
| 1024 | `prefill-direct` | 1..8 | 9,216 | eager | [prefill-direct Input 1024 TP=2](prefill-direct-metax-c500-tp2-qwen3-8b-input1024-batch1-8-eager-20261118.md) |
| 1024 | `decode-direct` | 1..256 | 9,216 | graph | [decode-direct Input 1024 TP=2](decode-direct-metax-c500-tp2-qwen3-8b-input1024-batch1-256-graph-20261118.md) |
| 2048 | `prefill-direct` | 1..8 | 18,432 | eager | [prefill-direct Input 2048 TP=2](prefill-direct-metax-c500-tp2-qwen3-8b-input2048-batch1-8-eager-20261118.md) |
| 2048 | `decode-direct` | 1..128 | 18,432 | graph | [decode-direct Input 2048 TP=2](decode-direct-metax-c500-tp2-qwen3-8b-input2048-batch1-128-graph-20261118.md) |
| 4096 | `prefill-direct` | 1..4 | 18,432 | eager | [prefill-direct Input 4096 TP=2](prefill-direct-metax-c500-tp2-qwen3-8b-input4096-batch1-4-eager-20261118.md) |
| 4096 | `decode-direct` | 1..64 | 18,432 | graph | [decode-direct Input 4096 TP=2](decode-direct-metax-c500-tp2-qwen3-8b-input4096-batch1-64-graph-20261118.md) |

## 核心结果

| Input | 模式 | 最佳 batch | 最佳吞吐 | TP=2 / TP=1 |
| ---: | --- | ---: | ---: | ---: |
| 512 | `prefill-direct` | 7 | 14,090.26 tok/s | 1.513x |
| 512 | `decode-direct` | 255 | 6,218.21 tok/s | 1.366x |
| 1024 | `prefill-direct` | 8 | 14,646.77 tok/s | 1.575x |
| 1024 | `decode-direct` | 256 | 5,257.30 tok/s | 1.538x |
| 2048 | `prefill-direct` | 8 | 14,450.36 tok/s | 1.582x |
| 2048 | `decode-direct` | 128 | 3,751.00 tok/s | 1.566x |
| 4096 | `prefill-direct` | 4 | 13,884.66 tok/s | 1.599x |
| 4096 | `decode-direct` | 64 | 2,163.46 tok/s | 1.615x |

TP=2 在 8 个组合中均提高总吞吐：prefill 为 1.513x 到 1.600x，decode 为 1.366x 到 1.615x。但 TP=2 占用两张 C500，且每卡峰值吞吐均低于 TP=1；从每卡效率看，TP=1 更高。

## 与 A100 TP=2 对比

| Input | 模式 | MetaX C500 TP=2 | A100 TP=2 | MetaX / A100 |
| ---: | --- | ---: | ---: | ---: |
| 512 | prefill | batch 7，14,090.26 tok/s | 21,452.86 tok/s | 0.657x |
| 512 | decode | batch 255，6,218.21 tok/s | 8,969.22 tok/s | 0.693x |
| 1024 | prefill | batch 8，14,646.77 tok/s | 21,804.24 tok/s | 0.672x |
| 1024 | decode | batch 256，5,257.30 tok/s | 7,552.53 tok/s | 0.696x |
| 2048 | prefill | batch 8，14,450.36 tok/s | 22,020.10 tok/s | 0.656x |
| 2048 | decode | batch 128，3,751.00 tok/s | 4,873.34 tok/s | 0.770x |
| 4096 | prefill | batch 4，13,884.66 tok/s | 21,294.20 tok/s | 0.652x |
| 4096 | decode | batch 64，2,163.46 tok/s | 2,731.98 tok/s | 0.792x |

两个平台使用不同硬件、驱动和 runtime，该表只作为量级参考，不是严格控制变量实验。MetaX input 1024 decode-direct 在 TP=2 下可完成 batch `1..256`，而 TP=1 在初始化 batch `1..256` 时发生 `mcMalloc` OOM，有效报告范围为 `1..192`。因此 input 1024 decode 的 TP=2 / TP=1 峰值对比包含 batch 范围差异。

另外，input 1024 的 MetaX TP=2 报告使用 `max_num_batched_tokens=9216`，A100 对应 TP 报告使用 `18432`；因此 input 1024 的 A100 对比除硬件、驱动和 runtime 差异外，还包含该配置差异。

## 原始产物

Raw JSONL 和 log 文件是本机运行产物，不随仓库提交。各报告保留了指向仓库根目录原始产物的相对路径，仅用于本机追溯。
