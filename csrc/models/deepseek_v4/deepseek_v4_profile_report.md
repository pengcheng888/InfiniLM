# DeepSeek V4 性能分析报告

## 概要

- 日期：2026-07-26
- 模型：`/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8`
- 运行目录：`/workspace_codex/InfiniLM`
- TP：8
- `MAX_NEW_TOKENS`：16
- CUDA Graph：未启用，`enable_graph=False`
- Attention：skip attention forward；`attn_hc_pre`、`attn_norm`、`attn_hc_post` 仍按模型主干执行
- MHC 算子：固定 kernel 路径
- Gate TopK：`INFINILM_DSV4_GATE_TOPK=kernel`
- Routed expert 后端：`INFINILM_DSV4_ROUTED_EXPERT_BACKEND=fused_experts_int8_marlin`
- Shared output 融合：`INFINILM_DSV4_FUSED_SHARED_OUTPUT=true`
- Profile 开关：`INFINILM_DSV4_PROFILE=1`，`INFINILM_DSV4_CAUSAL_DETAIL_PROFILE=1`
- Startup warmup：未启用；本文 profile 统计只包含正式生成阶段
- 原始日志：`csrc/models/deepseek_v4/profile_logs/deepseek_v4_full_tp8_nograph_kernel_skipattn_nowarmup_20260726.log`

## 运行命令

本次为了满足“无 graph”要求，使用 `run_infer.sh` 的等价参数手动展开，并去掉 `--enable-graph` 和 `--warmup`。

```bash
source ~/.bashrc
source /.myenv.sh
source ~/.xmake/profile
cd /workspace_codex/InfiniLM

export INFINILM_DSV4_PROFILE=1
export INFINILM_DSV4_CAUSAL_DETAIL_PROFILE=1
export INFINILM_DSV4_ROUTED_EXPERT_BACKEND=fused_experts_int8_marlin
export INFINILM_DSV4_FUSED_SHARED_OUTPUT=true
export INFINILM_DSV4_GATE_TOPK=kernel

python examples/test_infer.py \
  --device hygon \
  --model=/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8 \
  --temperature 1.0 \
  --top-p 0.8 \
  --top-k 1 \
  --tp 8 \
  --max-new-tokens 16 \
  --enable-paged-attn \
  --attn paged-attn
```

## 输出结果

- 权重加载耗时：`75705.178 ms`
- 输入 token id：`[0, 128803, 4117, 477, 440, 128804, 128822]`
- 输出 token id：`[16, 7249, 28, 1313, 223, 20, 69, 7234, 14, 20251, 760, 11, 940, 274, 63, 72]`
- 脚本生成总耗时：`2983.17 ms`

## Model Runner 阶段耗时

| 阶段 | step 数 | tokens | build 总耗时 ms | build 平均耗时 ms | forward 总耗时 ms | forward 平均耗时 ms | total 总耗时 ms | total 平均耗时 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill | 1 | 7 | 6.234 | 6.234 | 1878.233 | 1878.233 | 1884.761 | 1884.761 |
| decode | 15 | 1/step | 24.184 | 1.612 | 1052.630 | 70.175 | 1078.402 | 71.893 |

### Decode 每步耗时

| step | build_ms | forward_ms | total_ms |
| ---: | ---: | ---: | ---: |
| 1 | 1.207 | 89.313 | 90.784 |
| 2 | 1.561 | 83.257 | 85.094 |
| 3 | 1.992 | 67.982 | 70.081 |
| 4 | 1.574 | 67.865 | 69.521 |
| 5 | 1.638 | 66.636 | 68.359 |
| 6 | 1.664 | 67.273 | 69.019 |
| 7 | 1.631 | 67.382 | 69.096 |
| 8 | 1.611 | 69.471 | 71.160 |
| 9 | 1.659 | 68.921 | 70.652 |
| 10 | 1.626 | 68.779 | 70.481 |
| 11 | 1.638 | 66.939 | 68.651 |
| 12 | 1.613 | 68.042 | 69.728 |
| 13 | 1.568 | 66.962 | 68.606 |
| 14 | 1.601 | 67.749 | 69.428 |
| 15 | 1.601 | 66.059 | 67.742 |

## Profile 总览

`INFINILM_DSV4_PROFILE` 是 GPU-synced wall time，会在计时点同步 stream。它适合观察模块热点，但会拉高端到端耗时。

| 阶段 | decoder.layer 调用次数 | decoder.layer 总耗时 ms | decoder.layer 平均耗时 ms | decoder.moe 总耗时 ms | moe.forward 总耗时 ms | moe.gate 总耗时 ms | moe.topk 总耗时 ms | moe.experts 总耗时 ms | moe.shared_experts 总耗时 ms | moe.allreduce 总耗时 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 5504 | 13858.523 | 2.517900 | 9607.956 | 9523.819 | 462.770 | 416.385 | 3440.087 | 2156.018 | 2739.732 |
| prefill | 344 | 5936.893 | 17.258410 | 4275.983 | 4271.429 | 171.274 | 139.957 | 1176.649 | 1031.116 | 1733.372 |
| decode | 5160 | 7921.630 | 1.535200 | 5331.973 | 5252.390 | 291.496 | 276.428 | 2263.438 | 1124.902 | 1006.360 |

## Prefill 详细数据

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 344 | 5936.893 | 17.258410 | 100.00% |
| `decoder.attn_hc_pre` | 344 | 300.728 | 0.874209 | 5.07% |
| `decoder.attn_norm` | 344 | 336.775 | 0.978997 | 5.67% |
| `decoder.attn_hc_post` | 344 | 88.132 | 0.256198 | 1.48% |
| `decoder.ffn_hc_pre` | 344 | 452.637 | 1.315805 | 7.62% |
| `decoder.ffn_norm` | 344 | 434.261 | 1.262387 | 7.31% |
| `decoder.moe` | 344 | 4275.983 | 12.430183 | 72.02% |
| `decoder.ffn_hc_post` | 344 | 17.030 | 0.049506 | 0.29% |
| `moe.forward` | 344 | 4271.429 | 12.416945 | 71.95% |
| `moe.gate` | 344 | 171.274 | 0.497890 | 2.88% |
| `moe.topk` | 344 | 139.957 | 0.406852 | 2.36% |
| `moe.experts` | 344 | 1176.649 | 3.420491 | 19.82% |
| `moe.experts.contiguous` | 344 | 17.622 | 0.051227 | 0.30% |
| `moe.experts.fused_call` | 344 | 1150.332 | 3.343988 | 19.38% |
| `moe.shared_experts` | 344 | 1031.116 | 2.997430 | 17.37% |
| `moe.allreduce` | 344 | 1733.372 | 5.038872 | 29.20% |

## Decode 详细数据

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 5160 | 7921.630 | 1.535200 | 100.00% |
| `decoder.attn_hc_pre` | 5160 | 707.597 | 0.137131 | 8.93% |
| `decoder.attn_norm` | 5160 | 242.027 | 0.046904 | 3.06% |
| `decoder.attn_hc_post` | 5160 | 208.142 | 0.040338 | 2.63% |
| `decoder.ffn_hc_pre` | 5160 | 612.123 | 0.118628 | 7.73% |
| `decoder.ffn_norm` | 5160 | 222.162 | 0.043055 | 2.80% |
| `decoder.moe` | 5160 | 5331.973 | 1.033328 | 67.31% |
| `decoder.ffn_hc_post` | 5160 | 244.087 | 0.047304 | 3.08% |
| `moe.forward` | 5160 | 5252.390 | 1.017905 | 66.30% |
| `moe.gate` | 5160 | 291.496 | 0.056491 | 3.68% |
| `moe.topk` | 5160 | 276.428 | 0.053571 | 3.49% |
| `moe.experts` | 5160 | 2263.438 | 0.438651 | 28.57% |
| `moe.experts.contiguous` | 5160 | 281.747 | 0.054602 | 3.56% |
| `moe.experts.fused_call` | 5160 | 1859.557 | 0.360379 | 23.47% |
| `moe.shared_experts` | 5160 | 1124.902 | 0.218004 | 14.20% |
| `moe.allreduce` | 5160 | 1006.360 | 0.195031 | 12.70% |

## 结论

1. 当前 no-graph、TP8、kernel 算子、skip-attn 配置下，正式生成 `16` 个 token 的脚本总耗时为 `2983.17 ms`。
2. `ModelRunner` 视角下，prefill 单次总耗时为 `1884.761 ms`；decode 共 `15` 步，平均每步 `71.893 ms`。
3. Profile 视角下，decode 阶段 `decoder.moe` 占 `decoder.layer` 的 `67.31%`，仍是最大热点。
4. decode 阶段 MoE 内部主要耗时为 `moe.experts`、`moe.shared_experts` 和 `moe.allreduce`，平均分别为 `0.438651 ms/layer-call`、`0.218004 ms/layer-call`、`0.195031 ms/layer-call`。
5. 本报告已清除旧的 2026-07-22 多后端历史数据，只保留 2026-07-26 当前版本的一组 no-graph TP8 kernel skip-attn 数据。
