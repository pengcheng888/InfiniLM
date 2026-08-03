# DeepSeek V4 性能分析报告

## 概要

- 日期：2026-07-31
- 模型：`/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8`
- 运行目录：`/workspace_codex/InfiniLM`
- TP：8
- `MAX_NEW_TOKENS`：16
- CUDA Graph：未启用，`enable_graph=False`
- Attention：完整 DeepSeek V4 attention 主路径，包含 no-compress、compress ratio 4、compress ratio 128 三类 decoder layer
- Paged KV Cache：启用，`num_blocks=512`
- Gate TopK：`INFINILM_DSV4_GATE_TOPK=kernel`
- Routed expert 后端：`INFINILM_DSV4_ROUTED_EXPERT_BACKEND=fused_experts_int8_marlin`
- Shared output 融合：固定启用
- Profile 开关：`INFINILM_DSV4_PROFILE=1`
- Startup warmup：启用，`prompt_tokens=7`，`max_new_tokens=2`
- 原始日志：`csrc/models/deepseek_v4/profile_logs/deepseek_v4_full_tp8_nograph_attention_fine_20260731.log`

注意：`INFINILM_DSV4_PROFILE` 是 GPU-synced wall time，会在计时点同步 stream。它适合观察模块热点，但会拉高端到端耗时。本文 `Profile` 表包含 startup warmup 和正式生成；`Model Runner` 表单独列出 startup warmup 与正式生成。

## 运行命令

本次通过 `run_infer.sh` 运行，并使用 `ENABLE_GRAPH=0` 去掉 `--enable-graph`。

```bash
source ~/.bashrc
source /.myenv.sh
source ~/.xmake/profile
cd /workspace_codex/InfiniLM

ENABLE_GRAPH=0 INFINILM_DSV4_PROFILE=1 bash run_infer.sh
```

等价推理参数：

```bash
python examples/test_infer.py \
  --device hygon \
  --model=/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8 \
  --temperature 1.0 \
  --top-p 0.8 \
  --top-k 1 \
  --tp 8 \
  --max-new-tokens 16 \
  --warmup \
  --enable-paged-attn \
  --attn paged-attn
```

## 输出结果

- 权重加载耗时：`109832.725 ms`
- Startup warmup 耗时：`4833.519 ms`
- 正式生成总耗时：`2390.96 ms`
- 日志确认：`LLMEngine initialized ... enable_graph=False`
- 日志中无 `--enable-graph`、`run device_graph` 或 `Fall back to eager` 记录

## Model Runner 阶段耗时

### Startup Warmup

| 阶段 | step 数 | tokens | build 总耗时 ms | build 平均耗时 ms | forward 总耗时 ms | forward 平均耗时 ms | total 总耗时 ms | total 平均耗时 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill | 1 | 7 | 0.946 | 0.946 | 4549.762 | 4549.762 | 4550.974 | 4550.974 |
| decode | 1 | 1 | 0.328 | 0.328 | 281.291 | 281.291 | 281.757 | 281.757 |

### 正式生成

| 阶段 | step 数 | tokens | build 总耗时 ms | build 平均耗时 ms | forward 总耗时 ms | forward 平均耗时 ms | total 总耗时 ms | total 平均耗时 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill | 1 | 7 | 0.857 | 0.857 | 147.190 | 147.190 | 148.238 | 148.238 |
| decode | 15 | 1/step | 3.886 | 0.259 | 2226.410 | 148.427 | 2231.518 | 148.768 |

### 正式 Decode 每步耗时

| step | build_ms | forward_ms | total_ms |
| ---: | ---: | ---: | ---: |
| 1 | 0.263 | 143.934 | 144.286 |
| 2 | 0.256 | 140.652 | 140.986 |
| 3 | 0.258 | 141.435 | 141.776 |
| 4 | 0.251 | 165.492 | 165.828 |
| 5 | 0.296 | 216.825 | 217.206 |
| 6 | 0.257 | 141.465 | 141.798 |
| 7 | 0.253 | 141.732 | 142.064 |
| 8 | 0.249 | 141.190 | 141.520 |
| 9 | 0.254 | 142.004 | 142.338 |
| 10 | 0.253 | 140.797 | 141.129 |
| 11 | 0.254 | 142.783 | 143.119 |
| 12 | 0.249 | 142.170 | 142.497 |
| 13 | 0.260 | 141.944 | 142.282 |
| 14 | 0.271 | 141.522 | 141.874 |
| 15 | 0.262 | 142.465 | 142.815 |

## Profile 总览

Profile 调用次数按 TP rank 聚合。当前 full model 有 43 层：no-compress 2 层、compress ratio 4 为 21 层、compress ratio 128 为 20 层。统计 calls 满足：

```text
overall forward calls = 144
prefill forward calls = 16
decode forward calls = 128
decoder.layer calls = forward calls * 43
```

其中 144 次 forward 包含 startup warmup 和正式生成。

| 阶段 | causal.forward 总耗时 ms | decoder.layer 总耗时 ms | decoder.layer 占 causal.forward | attention.forward 总耗时 ms | attention.forward 占 decoder.layer | decoder.moe 总耗时 ms | decoder.moe 占 decoder.layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 57272.879 | 53938.007 | 94.18% | 38709.923 | 71.77% | 10792.080 | 20.01% |
| prefill | 37391.495 | 34267.817 | 91.65% | 27792.245 | 81.10% | 4916.898 | 14.35% |
| decode | 19881.384 | 19670.190 | 98.94% | 10917.678 | 55.50% | 5875.182 | 29.87% |

## Decoder Layer 类型分桶

百分比列以同一行的 `decoder.layer total_ms` 为分母。

### Overall

| layer type | 层数 | decoder.layer calls | decoder.layer total_ms | decoder.layer avg_ms | attention.forward total_ms | attention.forward avg_ms | attention 占比 | decoder.moe total_ms | decoder.moe avg_ms | MoE 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_compress | 2 | 288 | 30181.537 | 104.797003 | 24874.691 | 86.370455 | 82.42% | 3972.816 | 13.794500 | 13.16% |
| compress_ratio_4 | 21 | 3024 | 13851.988 | 4.580684 | 8847.406 | 2.925729 | 63.87% | 3368.475 | 1.113914 | 24.32% |
| compress_ratio_128 | 20 | 2880 | 9904.482 | 3.439056 | 4987.826 | 1.731884 | 50.36% | 3450.789 | 1.198191 | 34.84% |

### Prefill

| layer type | decoder.layer calls | decoder.layer total_ms | decoder.layer avg_ms | attention.forward total_ms | attention.forward avg_ms | attention 占比 | decoder.moe total_ms | decoder.moe avg_ms | MoE 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_compress | 32 | 29446.681 | 920.208781 | 24534.540 | 766.704375 | 83.32% | 3704.553 | 115.767281 | 12.58% |
| compress_ratio_4 | 336 | 3023.102 | 8.997327 | 2339.742 | 6.963518 | 77.40% | 504.147 | 1.500437 | 16.68% |
| compress_ratio_128 | 320 | 1798.034 | 5.618856 | 917.963 | 2.868634 | 51.05% | 708.198 | 2.213119 | 39.39% |

### Decode

| layer type | decoder.layer calls | decoder.layer total_ms | decoder.layer avg_ms | attention.forward total_ms | attention.forward avg_ms | attention 占比 | decoder.moe total_ms | decoder.moe avg_ms | MoE 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_compress | 256 | 734.856 | 2.870531 | 340.151 | 1.328715 | 46.29% | 268.263 | 1.047902 | 36.51% |
| compress_ratio_4 | 2688 | 10828.886 | 4.028603 | 6507.664 | 2.421006 | 60.10% | 2864.328 | 1.065598 | 26.45% |
| compress_ratio_128 | 2560 | 8106.448 | 3.166581 | 4069.863 | 1.589790 | 50.21% | 2742.591 | 1.071325 | 33.83% |

## Attention Decode 细粒度分桶

百分比列以同一 layer type 的 `attention.forward total_ms` 为分母。`attention.flashmla` 是外层计时，`attention.flashmla_workspace`、`attention.flashmla_out_workspace_call`、`attention.flashmla_with_metadata_call` 是其内部子项，不能与外层简单相加作为总耗时。

### No Compress

| 事件 | calls | total_ms | avg_ms | 占 attention.forward |
| --- | ---: | ---: | ---: | ---: |
| `attention.q_proj_a` | 256 | 37.917 | 0.148113 | 11.15% |
| `attention.q_norm` | 256 | 9.557 | 0.037332 | 2.81% |
| `attention.q_proj_b` | 256 | 24.172 | 0.094422 | 7.11% |
| `attention.q_rmsnorm_self` | 256 | 8.731 | 0.034105 | 2.57% |
| `attention.kv_proj` | 256 | 24.603 | 0.096105 | 7.23% |
| `attention.kv_norm` | 256 | 9.173 | 0.035832 | 2.70% |
| `attention.rope` | 256 | 13.404 | 0.052359 | 3.94% |
| `attention.swa_store` | 256 | 10.229 | 0.039957 | 3.01% |
| `attention.flashmla_schedule` | 256 | 1.089 | 0.004254 | 0.32% |
| `attention.flashmla_workspace` | 128 | 17.618 | 0.137641 | 5.18% |
| `attention.flashmla_out_workspace_call` | 128 | 16.950 | 0.132422 | 4.98% |
| `attention.flashmla_with_metadata_call` | 128 | 25.531 | 0.199461 | 7.51% |
| `attention.flashmla` | 256 | 63.471 | 0.247934 | 18.66% |
| `attention.out_rope` | 256 | 9.817 | 0.038348 | 2.89% |
| `attention.wo_a` | 256 | 19.147 | 0.074793 | 5.63% |
| `attention.wo_b` | 256 | 78.396 | 0.306234 | 23.05% |

### Compress Ratio 4

| 事件 | calls | total_ms | avg_ms | 占 attention.forward |
| --- | ---: | ---: | ---: | ---: |
| `attention.q_proj_a` | 2688 | 299.284 | 0.111341 | 4.60% |
| `attention.q_norm` | 2688 | 100.638 | 0.037440 | 1.55% |
| `attention.q_proj_b` | 2688 | 262.342 | 0.097597 | 4.03% |
| `attention.q_rmsnorm_self` | 2688 | 90.909 | 0.033820 | 1.40% |
| `attention.kv_proj` | 2688 | 257.846 | 0.095925 | 3.96% |
| `attention.kv_norm` | 2688 | 97.563 | 0.036296 | 1.50% |
| `attention.rope` | 2688 | 138.116 | 0.051382 | 2.12% |
| `attention.swa_store` | 2688 | 104.984 | 0.039057 | 1.61% |
| `attention.c4_compress` | 2688 | 696.663 | 0.259175 | 10.71% |
| `attention.c4_sparse_alloc` | 2688 | 18.405 | 0.006847 | 0.28% |
| `attention.c4_indexer.compress` | 2688 | 711.138 | 0.264560 | 10.93% |
| `attention.c4_indexer.query` | 2688 | 585.789 | 0.217927 | 9.00% |
| `attention.c4_indexer.sparse` | 2688 | 492.960 | 0.183393 | 7.58% |
| `attention.flashmla_schedule` | 2688 | 12.955 | 0.004820 | 0.20% |
| `attention.flashmla_workspace` | 2560 | 320.845 | 0.125330 | 4.93% |
| `attention.flashmla_out_workspace_call` | 2560 | 514.811 | 0.201098 | 7.91% |
| `attention.flashmla_with_metadata_call` | 128 | 27.098 | 0.211703 | 0.42% |
| `attention.flashmla` | 2688 | 893.898 | 0.332551 | 13.74% |
| `attention.out_rope` | 2688 | 104.241 | 0.038780 | 1.60% |
| `attention.wo_a` | 2688 | 242.419 | 0.090186 | 3.73% |
| `attention.wo_b` | 2688 | 977.944 | 0.363818 | 15.03% |

### Compress Ratio 128

| 事件 | calls | total_ms | avg_ms | 占 attention.forward |
| --- | ---: | ---: | ---: | ---: |
| `attention.q_proj_a` | 2560 | 282.493 | 0.110349 | 6.94% |
| `attention.q_norm` | 2560 | 94.251 | 0.036817 | 2.32% |
| `attention.q_proj_b` | 2560 | 249.687 | 0.097534 | 6.14% |
| `attention.q_rmsnorm_self` | 2560 | 86.604 | 0.033830 | 2.13% |
| `attention.kv_proj` | 2560 | 244.036 | 0.095327 | 6.00% |
| `attention.kv_norm` | 2560 | 92.971 | 0.036317 | 2.28% |
| `attention.rope` | 2560 | 132.622 | 0.051805 | 3.26% |
| `attention.swa_store` | 2560 | 100.857 | 0.039397 | 2.48% |
| `attention.c128_compress` | 2560 | 645.354 | 0.252091 | 15.86% |
| `attention.flashmla_schedule` | 2560 | 12.048 | 0.004706 | 0.30% |
| `attention.flashmla_workspace` | 2432 | 307.877 | 0.126594 | 7.56% |
| `attention.flashmla_out_workspace_call` | 2432 | 397.387 | 0.163399 | 9.76% |
| `attention.flashmla_with_metadata_call` | 128 | 29.189 | 0.228039 | 0.72% |
| `attention.flashmla` | 2560 | 764.228 | 0.298527 | 18.78% |
| `attention.out_rope` | 2560 | 101.866 | 0.039791 | 2.50% |
| `attention.wo_a` | 2560 | 188.426 | 0.073604 | 4.63% |
| `attention.wo_b` | 2560 | 740.589 | 0.289293 | 18.20% |

## MoE Decode 子项分桶

百分比列以同一 layer type 的 `decoder.moe total_ms` 为分母。

| layer type | moe.forward avg_ms | moe.gate avg_ms | gate 占比 | moe.topk avg_ms | topk 占比 | moe.experts avg_ms | experts 占比 | moe.shared_experts avg_ms | shared 占比 | moe.allreduce avg_ms | allreduce 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_compress | 1.037090 | 0.047102 | 4.49% | 0.034766 | 3.32% | 0.515590 | 49.20% | 0.232559 | 22.19% | 0.163031 | 15.56% |
| compress_ratio_4 | 1.054423 | 0.074609 | 7.00% | 0.050382 | 4.73% | 0.495732 | 46.52% | 0.240935 | 22.61% | 0.148769 | 13.96% |
| compress_ratio_128 | 1.060292 | 0.047565 | 4.44% | 0.049011 | 4.57% | 0.521138 | 48.64% | 0.230646 | 21.53% | 0.168045 | 15.69% |

## 结论

1. 当前 no-graph、TP8、完整 attention、profile 开启配置下，正式生成 `16` 个 token 的脚本总耗时为 `2390.96 ms`。
2. 正式生成 `ModelRunner` 视角下，prefill 总耗时为 `148.238 ms`；decode 共 `15` 步，平均每步 `148.768 ms`。
3. Profile 视角下，decode 阶段三类 decoder layer 的平均耗时分别为：no-compress `2.870531 ms`，compress ratio 4 `4.028603 ms`，compress ratio 128 `3.166581 ms`。
4. Decode 阶段 attention 平均耗时排序为：compress ratio 4 `2.421006 ms` > compress ratio 128 `1.589790 ms` > no-compress `1.328715 ms`。
5. Compress ratio 4 的 decode attention 中，C4 compress + C4 indexer compress/query/sparse 合计占 `38.22%` 的 `attention.forward`，是该类型 attention 变慢的主要来源。
6. Compress ratio 128 的 decode attention 中，`c128_compress` 占 `15.86%`，`flashmla` 占 `18.78%`，`wo_b` 占 `18.20%`。
7. No-compress decode attention 中，`wo_b` 占 `23.05%`，`flashmla` 占 `18.66%`，q/kv projection 相关子项合计约 `33.57%`。
8. Decode 阶段 MoE 三类 layer 的平均耗时接近，约 `1.05-1.07 ms/layer-call`；差异主要来自 attention 路径。
