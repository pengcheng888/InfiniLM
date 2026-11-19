# MetaX C500 TP=4 direct benchmark 报告索引

本目录收录 MetaX C500、TP=4 上的 `prefill-direct` 和 `decode-direct` 无干扰性能报告。全部报告使用 Qwen3-8B、flash-attn、paged KV cache、warmup 3 次、正式测量 30 次 repeat。Input 1024 使用 `max_num_batched_tokens=9216`，其余组合使用 18432。

## 测试环境

| 项目 | 值 |
| --- | --- |
| 平台 | MetaX C500，64 GiB |
| Driver / runtime | MetaX driver 3.3.12 / MACA 3.2.1.10 |
| 设备 | 物理专用 GPU 3、4、5、6 |
| 模型 | `/data-aisoft/mechdancer/models/Qwen3-8B` |
| Attention | `flash-attn` |
| Tensor parallel | 4 |
| Warmup / 正式 repeat | 每形状 3 次 / 30 次 |
| Decode step | 每 repeat 4 step |
| 干扰控制 | 每 10 秒采样 `mx-smi` |

## 干扰检查

监控文件：`benchmark-metax-c500-tp4-gpu-monitor-20261118.log`。

- 监控间隔：10 秒。
- 采样点：231 个。
- 时间范围：2026-11-18 18:31:16 - 2026-11-18 19:10:27。
- GPU 3/4/5/6 外来进程采样点：0。
- GPU 3/4/5/6 多进程采样点：0。
- GPU 3/4/5/6 进程显存峰值：20,058 MiB。
- GPU 3/4/5/6 温度范围：30C-54C。
- GPU 3/4/5/6 功耗范围：92W-307W。
- 测试结束后 GPU 3/4/5/6 无残留 benchmark 进程。

## 测试脚本说明

TP=4 报告产生时，本地 `run_bench-metax.sh` 硬编码 `--tp 4`，并使用物理 GPU 3、4、5、6。该脚本被 `.gitignore` 忽略，不随仓库提交；单项报告中的命令是历史调用记录。复现前需确认脚本仍为 `--tp 4`，并使用上述四张专用 GPU。

## 报告索引

| Input length | 模式 | Batch range | Max batched tokens | 执行模式 | 报告 |
| ---: | --- | --- | ---: | --- | --- |
| 512 | `prefill-direct` | 1..8 | 18,432 | eager | [prefill-direct Input 512 TP=4](prefill-direct-metax-c500-tp4-qwen3-8b-input512-batch1-8-eager-20261118.md) |
| 512 | `decode-direct` | 1..256 | 18,432 | graph | [decode-direct Input 512 TP=4](decode-direct-metax-c500-tp4-qwen3-8b-input512-batch1-256-graph-20261118.md) |
| 1024 | `prefill-direct` | 1..8 | 9,216 | eager | [prefill-direct Input 1024 TP=4](prefill-direct-metax-c500-tp4-qwen3-8b-input1024-batch1-8-eager-20261118.md) |
| 1024 | `decode-direct` | 1..256 | 9,216 | graph | [decode-direct Input 1024 TP=4](decode-direct-metax-c500-tp4-qwen3-8b-input1024-batch1-256-graph-20261118.md) |
| 2048 | `prefill-direct` | 1..8 | 18,432 | eager | [prefill-direct Input 2048 TP=4](prefill-direct-metax-c500-tp4-qwen3-8b-input2048-batch1-8-eager-20261118.md) |
| 2048 | `decode-direct` | 1..128 | 18,432 | graph | [decode-direct Input 2048 TP=4](decode-direct-metax-c500-tp4-qwen3-8b-input2048-batch1-128-graph-20261118.md) |
| 4096 | `prefill-direct` | 1..4 | 18,432 | eager | [prefill-direct Input 4096 TP=4](prefill-direct-metax-c500-tp4-qwen3-8b-input4096-batch1-4-eager-20261118.md) |
| 4096 | `decode-direct` | 1..64 | 18,432 | graph | [decode-direct Input 4096 TP=4](decode-direct-metax-c500-tp4-qwen3-8b-input4096-batch1-64-graph-20261118.md) |

## 核心结果

| Input | 模式 | 最佳 batch | 最佳吞吐 | TP=4 / TP=2 | TP=4 / TP=1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | `prefill-direct` | 8 | 24,617.17 tok/s | 1.747x | 2.644x |
| 512 | `decode-direct` | 255 | 9,417.38 tok/s | 1.514x | 2.069x |
| 1024 | `prefill-direct` | 8 | 25,917.55 tok/s | 1.770x | 2.787x |
| 1024 | `decode-direct` | 255 | 8,366.76 tok/s | 1.591x | 2.447x |
| 2048 | `prefill-direct` | 8 | 25,896.44 tok/s | 1.792x | 2.835x |
| 2048 | `decode-direct` | 128 | 5,665.03 tok/s | 1.510x | 2.366x |
| 4096 | `prefill-direct` | 4 | 25,041.42 tok/s | 1.804x | 2.885x |
| 4096 | `decode-direct` | 63 | 3,304.40 tok/s | 1.527x | 2.467x |

TP=4 相比 TP=2 继续提高总吞吐：prefill 为 1.747x 到 1.804x，decode 为 1.510x 到 1.591x。但 TP=4 占用四张 C500，每卡效率明显低于 TP=2 和 TP=1。

## 与 A100 TP=4 对比

| Input | 模式 | MetaX C500 TP=4 | A100 TP=4 | MetaX / A100 |
| ---: | --- | ---: | ---: | ---: |
| 512 | prefill | batch 8，24,617.17 tok/s | 34,781.56 tok/s | 0.708x |
| 512 | decode | batch 255，9,417.38 tok/s | 12,317.88 tok/s | 0.765x |
| 1024 | prefill | batch 8，25,917.55 tok/s | 35,960.64 tok/s | 0.721x |
| 1024 | decode | batch 255，8,366.76 tok/s | 10,683.02 tok/s | 0.783x |
| 2048 | prefill | batch 8，25,896.44 tok/s | 36,634.03 tok/s | 0.707x |
| 2048 | decode | batch 128，5,665.03 tok/s | 7,058.49 tok/s | 0.803x |
| 4096 | prefill | batch 4，25,041.42 tok/s | 35,606.70 tok/s | 0.703x |
| 4096 | decode | batch 63，3,304.40 tok/s | 4,181.42 tok/s | 0.790x |

两个平台使用不同硬件、驱动和 runtime，该表只作为量级参考，不是严格控制变量实验。MetaX input 1024 decode-direct 在 TP=2 和 TP=4 下均可完成 batch `1..256`，而 TP=1 在初始化 batch `1..256` 时发生 `mcMalloc` OOM，有效报告范围为 `1..192`。因此 input 1024 decode 的 TP=4 / TP=1 峰值对比包含 batch 范围差异。

另外，input 1024 的 MetaX TP=4 报告使用 `max_num_batched_tokens=9216`，A100 对应 TP 报告使用 `18432`；因此 input 1024 的 A100 对比除硬件、驱动和 runtime 差异外，还包含该配置差异。

## 原始产物

Raw JSONL 和 log 文件是本机运行产物，不随仓库提交。各报告保留了指向仓库根目录原始产物的相对路径，仅用于本机追溯。
