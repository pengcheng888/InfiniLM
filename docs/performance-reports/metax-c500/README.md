# MetaX C500 direct benchmark 报告索引

本目录收录 MetaX C500 上 `prefill-direct` 和 `decode-direct` 的无干扰重测报告。全部报告使用 Qwen3-8B、flash-attn、paged KV cache 和物理 GPU 5。

## 测试环境

| 项目 | 值 |
| --- | --- |
| 平台 | MetaX C500，64 GiB |
| Driver / runtime | MetaX driver 3.3.12 / MACA 3.2.1.10 |
| 模型 | `/data-aisoft/mechdancer/models/Qwen3-8B` |
| Attention | `flash-attn` |
| Tensor parallel | 1 |
| Warmup / 正式 repeat | 每形状 3 次 / 30 次 |
| Decode step | 每 repeat 4 step |
| 干扰控制 | 物理专用 GPU 5；每 10 秒采样 `mx-smi` |

## 报告索引

| Input length | 模式 | Batch range | Max batched tokens | 执行模式 | 报告 |
| ---: | --- | --- | ---: | --- | --- |
| 512 | `prefill-direct` | 1..8 | 18,432 | eager | [prefill-direct-metax-c500-qwen3-8b-input512-batch1-8-eager-20261113](prefill-direct-metax-c500-qwen3-8b-input512-batch1-8-eager-20261113.md) |
| 1024 | `prefill-direct` | 1..8 | 9,216 | eager | [prefill-direct-metax-c500-qwen3-8b-input1024-batch1-8-eager-20261113](prefill-direct-metax-c500-qwen3-8b-input1024-batch1-8-eager-20261113.md) |
| 2048 | `prefill-direct` | 1..8 | 18,432 | eager | [prefill-direct-metax-c500-qwen3-8b-input2048-batch1-8-eager-20261113](prefill-direct-metax-c500-qwen3-8b-input2048-batch1-8-eager-20261113.md) |
| 4096 | `prefill-direct` | 1..4 | 18,432 | eager | [prefill-direct-metax-c500-qwen3-8b-input4096-batch1-4-eager-20261113](prefill-direct-metax-c500-qwen3-8b-input4096-batch1-4-eager-20261113.md) |
| 512 | `decode-direct` | 1..256 | 18,432 | graph | [decode-direct-metax-c500-qwen3-8b-input512-batch1-256-graph-20261113](decode-direct-metax-c500-qwen3-8b-input512-batch1-256-graph-20261113.md) |
| 1024 | `decode-direct` | 1..192 | 9,216 | graph | [decode-direct-metax-c500-qwen3-8b-input1024-batch1-192-graph-20261113](decode-direct-metax-c500-qwen3-8b-input1024-batch1-192-graph-20261113.md) |
| 2048 | `decode-direct` | 1..128 | 18,432 | graph | [decode-direct-metax-c500-qwen3-8b-input2048-batch1-128-graph-20261113](decode-direct-metax-c500-qwen3-8b-input2048-batch1-128-graph-20261113.md) |
| 4096 | `decode-direct` | 1..64 | 18,432 | graph | [decode-direct-metax-c500-qwen3-8b-input4096-batch1-64-graph-20261113](decode-direct-metax-c500-qwen3-8b-input4096-batch1-64-graph-20261113.md) |

## 干扰检查

监控文件：`benchmark-clean-gpu-monitor-20261113-0753.log`。

- 监控间隔：10 秒。
- 采样点：575 个。
- 时间范围：2026-11-13 15:53:32-17:31:55（Asia/Shanghai）。
- GPU 5 外来进程采样点：0。
- GPU 5 多进程采样点：0。
- GPU 5 进程显存峰值：63,536 MiB。
- GPU 5 温度范围：32C-55C。
- GPU 5 功耗范围：91W-317W。

GPU 0-3 曾出现其他用户进程，但没有使用专用 GPU 5。测试结束后所有 benchmark 进程均已退出。

## 核心结果

| Input | 模式 | 最佳 batch | 最佳吞吐 |
| ---: | --- | ---: | ---: |
| 512 | `prefill-direct` | 7 | 9,310.41 tok/s |
| 1024 | `prefill-direct` | 7 | 9,298.77 tok/s |
| 2048 | `prefill-direct` | 8 | 9,135.67 tok/s |
| 4096 | `prefill-direct` | 4 | 8,681.30 tok/s |
| 512 | `decode-direct` | 256 | 4,552.35 tok/s |
| 1024 | `decode-direct` | 192 | 3,418.89 tok/s |
| 2048 | `decode-direct` | 104 | 2,394.80 tok/s |
| 4096 | `decode-direct` | 64 | 1,339.42 tok/s |

## OOM 说明

Input 1024 decode-direct 按原 A100 范围 batch `1..256` 在独立 C500 GPU 5 上初始化阶段发生 `mcMalloc` OOM。因此 C500 报告使用 batch `1..192`；OOM log 保留在仓库根目录。

## 原始产物

原始 JSONL 和 log 均为本机运行产物，不随仓库提交。clean 数据文件使用 `*-clean-20261113.jsonl` / `*.log` 后缀，位于仓库根目录。
