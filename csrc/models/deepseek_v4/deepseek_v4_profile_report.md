# DeepSeek V4 完整模型性能分析报告

## 概要

- 日期：2026-07-22
- 模型：`/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8`
- 启动脚本：`/workspace_codex/InfiniLM/run_infer.sh`
- TP：8
- `MAX_NEW_TOKENS`：16
- 性能分析开关：`INFINILM_DSV4_PROFILE=1`
- MHC 算子：`INFINILM_DSV4_MHC_PRE=kernel`，`INFINILM_DSV4_MHC_POST=kernel`，`INFINILM_DSV4_MHC_HEAD=kernel`
- 路由专家后端：`naive`、`lmslim_fused`、`aiter_split`、`lightop_split`
- 运行状态：四种后端均完成完整权重加载和 16 token 生成

## 运行命令

四种后端只替换 `INFINILM_DSV4_ROUTED_EXPERT_BACKEND` 的值，其余设置保持一致：

```bash
INFINILM_DSV4_PROFILE=1 \
INFINILM_DSV4_ROUTED_EXPERT_BACKEND=<backend> \
INFINILM_DSV4_MHC_POST=kernel \
INFINILM_DSV4_MHC_PRE=kernel \
INFINILM_DSV4_MHC_HEAD=kernel \
bash run_infer.sh
```

## 原始日志

- `naive`：`csrc/models/deepseek_v4/profile_logs/deepseek_v4_full_tp8_naive_profile_20260722.log`
- `lmslim_fused`：`csrc/models/deepseek_v4/profile_logs/deepseek_v4_full_tp8_lmslim_fused_profile_20260722.log`
- `aiter_split`：`csrc/models/deepseek_v4/profile_logs/deepseek_v4_full_tp8_aiter_split_profile_20260722.log`
- `lightop_split`：`csrc/models/deepseek_v4/profile_logs/deepseek_v4_full_tp8_lightop_profile_20260722.log`

## 输出结果

四次运行的输入 token id 相同：`[0, 128803, 4117, 477, 440, 128804, 128822]`。

| 后端 | 权重加载耗时 ms | 脚本生成总耗时 ms | 相对 naive 生成加速比 | 生成 token id |
| --- | ---: | ---: | ---: | --- |
| `naive` | 68020.223 | 33538.510 | 1.00x | `[344, 35, 109254, 343, 70722, 82454, 1189, 343, 70722, 82454, 1189, 343, 70722, 82454, 1189, 343]` |
| `lmslim_fused` | 70177.941 | 12319.180 | 2.72x | `[28, 126759, 46, 11173, 36, 94, 223, 5, 223, 5, 223, 5, 223, 5, 223, 5]` |
| `aiter_split` | 78616.945 | 6303.520 | 5.32x | `[965, 361, 51564, 123780, 104017, 51616, 35681, 10856, 117465, 57648, 122445, 54639, 100753, 97466, 97466, 97466]` |
| `lightop_split` | 67848.002 | 7846.430 | 4.27x | `[28, 126759, 362, 756, 1867, 79161, 442, 10461, 28, 126759, 362, 756, 1867, 79161, 442, 10461]` |

注意：四种后端的生成 token id 不完全一致，因此本报告只对当前运行的性能数据做横向比较；精度与 token 对齐仍需要单独按 SGLang attention-off 标准答案验证。

## 性能总览

以下加速比均以 `naive` 为基线；性能分析计时器会在每个计时代码块前后同步 GPU stream，因此适合观察相对热点。

### 总体阶段

| 后端 | decoder.layer 总耗时 ms | 相对 naive | decoder.moe 总耗时 ms | moe.topk 总耗时 ms | moe.experts 总耗时 ms | moe.allreduce 总耗时 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive` | 260355.197 | 1.00x | 256266.315 | 21697.444 | 219710.875 | 11474.402 |
| `lmslim_fused` | 89833.994 | 2.90x | 83547.067 | 20864.462 | 53635.843 | 5659.756 |
| `aiter_split` | 41841.577 | 6.22x | 37445.216 | 21069.355 | 9734.263 | 3209.087 |
| `lightop_split` | 53837.978 | 4.84x | 49726.592 | 22811.113 | 18730.869 | 4460.991 |

### 预填充阶段

| 后端 | decoder.layer 总耗时 ms | 相对 naive | decoder.moe 总耗时 ms | moe.topk 总耗时 ms | moe.experts 总耗时 ms | moe.allreduce 总耗时 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive` | 45902.824 | 1.00x | 45256.405 | 19720.431 | 22714.504 | 1680.418 |
| `lmslim_fused` | 29069.904 | 1.58x | 28147.625 | 18424.450 | 6754.662 | 1666.263 |
| `aiter_split` | 24762.405 | 1.85x | 23691.120 | 19096.854 | 2544.742 | 688.490 |
| `lightop_split` | 27533.915 | 1.67x | 26578.679 | 20868.684 | 2693.175 | 1251.206 |

### 解码阶段

| 后端 | decoder.layer 总耗时 ms | 相对 naive | decoder.moe 总耗时 ms | moe.topk 总耗时 ms | moe.experts 总耗时 ms | moe.allreduce 总耗时 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive` | 214452.373 | 1.00x | 211009.910 | 1977.013 | 196996.371 | 9793.984 |
| `lmslim_fused` | 60764.090 | 3.53x | 55399.442 | 2440.012 | 46881.181 | 3993.493 |
| `aiter_split` | 17079.172 | 12.56x | 13754.096 | 1972.501 | 7189.521 | 2520.597 |
| `lightop_split` | 26304.063 | 8.15x | 23147.913 | 1942.429 | 16037.694 | 3209.785 |

## 对比结论

按脚本生成总耗时排序：`aiter_split`、`lightop_split`、`lmslim_fused`、`naive`。
按性能分析的总体 `decoder.layer` 总耗时排序：`aiter_split`、`lightop_split`、`lmslim_fused`、`naive`。
预填充阶段 `moe.topk` 在四种后端中都占比较高，短 prompt 下路由/topk 是主要热点之一。
解码阶段差异主要来自 `moe.experts`：`naive` 为 `196996.371 ms`，`aiter_split` 为 `7189.521 ms`，该项约 `27.40x`。
`lmslim_fused` 相比 `naive` 明显加速；本次 `lightop_split` 和 `aiter_split` 的分步路径更快，其中 `aiter_split` 脚本生成总耗时最快，为 `6303.520 ms`。
`aiter_split` 日志中大量出现 `No matching kernel configuration found, using default settings`，说明当前 shape 没有命中特化配置，后续仍应补齐 mode 表或减少重复告警输出。
`lmslim_fused`、`aiter_split`、`lightop_split` 均出现 `moe_align_block_size_kernel` launch bounds 警告；本次运行未失败，但该 kernel 的 launch 配置仍值得后续整理。

## 各后端详细数据

### `naive`

- 说明：naive 参考后端
- 权重加载耗时：`68020.223 ms`
- 脚本生成总耗时：`33538.510 ms`
- 生成 token id：`[344, 35, 109254, 343, 70722, 82454, 1189, 343, 70722, 82454, 1189, 343, 70722, 82454, 1189, 343]`

#### 总体阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 5504 | 260355.197 | 47.303 | 100.00% |
| `decoder.attn_hc_pre` | 5504 | 1146.351 | 0.208 | 0.44% |
| `decoder.attn_norm` | 5504 | 347.863 | 0.063 | 0.13% |
| `decoder.attn_hc_post` | 5504 | 389.283 | 0.071 | 0.15% |
| `decoder.ffn_hc_pre` | 5504 | 970.283 | 0.176 | 0.37% |
| `decoder.ffn_norm` | 5504 | 357.766 | 0.065 | 0.14% |
| `decoder.moe` | 5504 | 256266.315 | 46.560 | 98.43% |
| `decoder.ffn_hc_post` | 5504 | 428.458 | 0.078 | 0.16% |
| `moe.forward` | 5504 | 256197.481 | 46.548 | 98.40% |
| `moe.topk` | 5504 | 21697.444 | 3.942 | 8.33% |
| `moe.experts` | 5504 | 219710.875 | 39.918 | 84.39% |
| `moe.shared_experts` | 5504 | 2620.445 | 0.476 | 1.01% |
| `moe.add_shared` | 5504 | 431.961 | 0.078 | 0.17% |
| `moe.allreduce` | 5504 | 11474.402 | 2.085 | 4.41% |

#### 预填充阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 344 | 45902.824 | 133.438 | 100.00% |
| `decoder.attn_hc_pre` | 344 | 147.127 | 0.428 | 0.32% |
| `decoder.attn_norm` | 344 | 79.254 | 0.230 | 0.17% |
| `decoder.attn_hc_post` | 344 | 86.114 | 0.250 | 0.19% |
| `decoder.ffn_hc_pre` | 344 | 155.682 | 0.453 | 0.34% |
| `decoder.ffn_norm` | 344 | 113.968 | 0.331 | 0.25% |
| `decoder.moe` | 344 | 45256.405 | 131.559 | 98.59% |
| `decoder.ffn_hc_post` | 344 | 32.389 | 0.094 | 0.07% |
| `moe.forward` | 344 | 45252.190 | 131.547 | 98.58% |
| `moe.topk` | 344 | 19720.431 | 57.327 | 42.96% |
| `moe.experts` | 344 | 22714.504 | 66.031 | 49.48% |
| `moe.shared_experts` | 344 | 1043.450 | 3.033 | 2.27% |
| `moe.add_shared` | 344 | 76.000 | 0.221 | 0.17% |
| `moe.allreduce` | 344 | 1680.418 | 4.885 | 3.66% |

#### 解码阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 5160 | 214452.373 | 41.561 | 100.00% |
| `decoder.attn_hc_pre` | 5160 | 999.224 | 0.194 | 0.47% |
| `decoder.attn_norm` | 5160 | 268.609 | 0.052 | 0.13% |
| `decoder.attn_hc_post` | 5160 | 303.169 | 0.059 | 0.14% |
| `decoder.ffn_hc_pre` | 5160 | 814.601 | 0.158 | 0.38% |
| `decoder.ffn_norm` | 5160 | 243.798 | 0.047 | 0.11% |
| `decoder.moe` | 5160 | 211009.910 | 40.893 | 98.39% |
| `decoder.ffn_hc_post` | 5160 | 396.069 | 0.077 | 0.18% |
| `moe.forward` | 5160 | 210945.291 | 40.881 | 98.36% |
| `moe.topk` | 5160 | 1977.013 | 0.383 | 0.92% |
| `moe.experts` | 5160 | 196996.371 | 38.178 | 91.86% |
| `moe.shared_experts` | 5160 | 1576.995 | 0.306 | 0.74% |
| `moe.add_shared` | 5160 | 355.961 | 0.069 | 0.17% |
| `moe.allreduce` | 5160 | 9793.984 | 1.898 | 4.57% |

### `lmslim_fused`

- 说明：lmslim fused Marlin 后端
- 权重加载耗时：`70177.941 ms`
- 脚本生成总耗时：`12319.180 ms`
- 生成 token id：`[28, 126759, 46, 11173, 36, 94, 223, 5, 223, 5, 223, 5, 223, 5, 223, 5]`
- 备注：出现 `moe_align_block_size_kernel` launch bounds 警告。

#### 总体阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 5504 | 89833.994 | 16.322 | 100.00% |
| `decoder.attn_hc_pre` | 5504 | 1101.989 | 0.200 | 1.23% |
| `decoder.attn_norm` | 5504 | 1075.476 | 0.195 | 1.20% |
| `decoder.attn_hc_post` | 5504 | 966.422 | 0.176 | 1.08% |
| `decoder.ffn_hc_pre` | 5504 | 1914.549 | 0.348 | 2.13% |
| `decoder.ffn_norm` | 5504 | 400.898 | 0.073 | 0.45% |
| `decoder.moe` | 5504 | 83547.067 | 15.179 | 93.00% |
| `decoder.ffn_hc_post` | 5504 | 369.929 | 0.067 | 0.41% |
| `moe.forward` | 5504 | 83481.141 | 15.167 | 92.93% |
| `moe.topk` | 5504 | 20864.462 | 3.791 | 23.23% |
| `moe.experts` | 5504 | 53635.843 | 9.745 | 59.71% |
| `moe.shared_experts` | 5504 | 2576.397 | 0.468 | 2.87% |
| `moe.add_shared` | 5504 | 480.804 | 0.087 | 0.54% |
| `moe.allreduce` | 5504 | 5659.756 | 1.028 | 6.30% |

#### 预填充阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 344 | 29069.904 | 84.506 | 100.00% |
| `decoder.attn_hc_pre` | 344 | 179.510 | 0.522 | 0.62% |
| `decoder.attn_norm` | 344 | 227.321 | 0.661 | 0.78% |
| `decoder.attn_hc_post` | 344 | 71.712 | 0.208 | 0.25% |
| `decoder.ffn_hc_pre` | 344 | 223.983 | 0.651 | 0.77% |
| `decoder.ffn_norm` | 344 | 162.501 | 0.472 | 0.56% |
| `decoder.moe` | 344 | 28147.625 | 81.824 | 96.83% |
| `decoder.ffn_hc_post` | 344 | 26.267 | 0.076 | 0.09% |
| `moe.forward` | 344 | 28143.487 | 81.812 | 96.81% |
| `moe.topk` | 344 | 18424.450 | 53.559 | 63.38% |
| `moe.experts` | 344 | 6754.662 | 19.636 | 23.24% |
| `moe.shared_experts` | 344 | 1129.536 | 3.284 | 3.89% |
| `moe.add_shared` | 344 | 151.352 | 0.440 | 0.52% |
| `moe.allreduce` | 344 | 1666.263 | 4.844 | 5.73% |

#### 解码阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 5160 | 60764.090 | 11.776 | 100.00% |
| `decoder.attn_hc_pre` | 5160 | 922.479 | 0.179 | 1.52% |
| `decoder.attn_norm` | 5160 | 848.155 | 0.164 | 1.40% |
| `decoder.attn_hc_post` | 5160 | 894.710 | 0.173 | 1.47% |
| `decoder.ffn_hc_pre` | 5160 | 1690.566 | 0.328 | 2.78% |
| `decoder.ffn_norm` | 5160 | 238.397 | 0.046 | 0.39% |
| `decoder.moe` | 5160 | 55399.442 | 10.736 | 91.17% |
| `decoder.ffn_hc_post` | 5160 | 343.662 | 0.067 | 0.57% |
| `moe.forward` | 5160 | 55337.654 | 10.724 | 91.07% |
| `moe.topk` | 5160 | 2440.012 | 0.473 | 4.02% |
| `moe.experts` | 5160 | 46881.181 | 9.085 | 77.15% |
| `moe.shared_experts` | 5160 | 1446.861 | 0.280 | 2.38% |
| `moe.add_shared` | 5160 | 329.452 | 0.064 | 0.54% |
| `moe.allreduce` | 5160 | 3993.493 | 0.774 | 6.57% |

### `aiter_split`

- 说明：AITER 拆分 Marlin 后端
- 权重加载耗时：`78616.945 ms`
- 脚本生成总耗时：`6303.520 ms`
- 生成 token id：`[965, 361, 51564, 123780, 104017, 51616, 35681, 10856, 117465, 57648, 122445, 54639, 100753, 97466, 97466, 97466]`
- 备注：出现 `moe_align_block_size_kernel` launch bounds 警告。
- 备注：出现 AITER Marlin 默认配置告警。

#### 总体阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 5504 | 41841.577 | 7.602 | 100.00% |
| `decoder.attn_hc_pre` | 5504 | 1108.232 | 0.201 | 2.65% |
| `decoder.attn_norm` | 5504 | 509.716 | 0.093 | 1.22% |
| `decoder.attn_hc_post` | 5504 | 423.159 | 0.077 | 1.01% |
| `decoder.ffn_hc_pre` | 5504 | 1008.389 | 0.183 | 2.41% |
| `decoder.ffn_norm` | 5504 | 411.995 | 0.075 | 0.98% |
| `decoder.moe` | 5504 | 37445.216 | 6.803 | 89.49% |
| `decoder.ffn_hc_post` | 5504 | 484.472 | 0.088 | 1.16% |
| `moe.forward` | 5504 | 37379.616 | 6.791 | 89.34% |
| `moe.topk` | 5504 | 21069.355 | 3.828 | 50.36% |
| `moe.experts` | 5504 | 9734.263 | 1.769 | 23.26% |
| `moe.shared_experts` | 5504 | 2627.414 | 0.477 | 6.28% |
| `moe.add_shared` | 5504 | 476.686 | 0.087 | 1.14% |
| `moe.allreduce` | 5504 | 3209.087 | 0.583 | 7.67% |

#### 预填充阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 344 | 24762.405 | 71.984 | 100.00% |
| `decoder.attn_hc_pre` | 344 | 193.994 | 0.564 | 0.78% |
| `decoder.attn_norm` | 344 | 246.212 | 0.716 | 0.99% |
| `decoder.attn_hc_post` | 344 | 108.329 | 0.315 | 0.44% |
| `decoder.ffn_hc_pre` | 344 | 192.037 | 0.558 | 0.78% |
| `decoder.ffn_norm` | 344 | 171.069 | 0.497 | 0.69% |
| `decoder.moe` | 344 | 23691.120 | 68.870 | 95.67% |
| `decoder.ffn_hc_post` | 344 | 127.483 | 0.371 | 0.51% |
| `moe.forward` | 344 | 23687.036 | 68.858 | 95.66% |
| `moe.topk` | 344 | 19096.854 | 55.514 | 77.12% |
| `moe.experts` | 344 | 2544.742 | 7.398 | 10.28% |
| `moe.shared_experts` | 344 | 1210.156 | 3.518 | 4.89% |
| `moe.add_shared` | 344 | 129.410 | 0.376 | 0.52% |
| `moe.allreduce` | 344 | 688.490 | 2.001 | 2.78% |

#### 解码阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 5160 | 17079.172 | 3.310 | 100.00% |
| `decoder.attn_hc_pre` | 5160 | 914.238 | 0.177 | 5.35% |
| `decoder.attn_norm` | 5160 | 263.504 | 0.051 | 1.54% |
| `decoder.attn_hc_post` | 5160 | 314.830 | 0.061 | 1.84% |
| `decoder.ffn_hc_pre` | 5160 | 816.352 | 0.158 | 4.78% |
| `decoder.ffn_norm` | 5160 | 240.926 | 0.047 | 1.41% |
| `decoder.moe` | 5160 | 13754.096 | 2.666 | 80.53% |
| `decoder.ffn_hc_post` | 5160 | 356.989 | 0.069 | 2.09% |
| `moe.forward` | 5160 | 13692.580 | 2.654 | 80.17% |
| `moe.topk` | 5160 | 1972.501 | 0.382 | 11.55% |
| `moe.experts` | 5160 | 7189.521 | 1.393 | 42.10% |
| `moe.shared_experts` | 5160 | 1417.258 | 0.275 | 8.30% |
| `moe.add_shared` | 5160 | 347.276 | 0.067 | 2.03% |
| `moe.allreduce` | 5160 | 2520.597 | 0.488 | 14.76% |

### `lightop_split`

- 说明：lightop 拆分 Marlin 后端
- 权重加载耗时：`67848.002 ms`
- 脚本生成总耗时：`7846.430 ms`
- 生成 token id：`[28, 126759, 362, 756, 1867, 79161, 442, 10461, 28, 126759, 362, 756, 1867, 79161, 442, 10461]`
- 备注：出现 `moe_align_block_size_kernel` launch bounds 警告。

#### 总体阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 5504 | 53837.978 | 9.782 | 100.00% |
| `decoder.attn_hc_pre` | 5504 | 1044.747 | 0.190 | 1.94% |
| `decoder.attn_norm` | 5504 | 396.259 | 0.072 | 0.74% |
| `decoder.attn_hc_post` | 5504 | 356.113 | 0.065 | 0.66% |
| `decoder.ffn_hc_pre` | 5504 | 1047.753 | 0.190 | 1.95% |
| `decoder.ffn_norm` | 5504 | 444.381 | 0.081 | 0.83% |
| `decoder.moe` | 5504 | 49726.592 | 9.035 | 92.36% |
| `decoder.ffn_hc_post` | 5504 | 362.605 | 0.066 | 0.67% |
| `moe.forward` | 5504 | 49655.267 | 9.022 | 92.23% |
| `moe.topk` | 5504 | 22811.113 | 4.144 | 42.37% |
| `moe.experts` | 5504 | 18730.869 | 3.403 | 34.79% |
| `moe.shared_experts` | 5504 | 2883.541 | 0.524 | 5.36% |
| `moe.add_shared` | 5504 | 509.616 | 0.093 | 0.95% |
| `moe.allreduce` | 5504 | 4460.991 | 0.810 | 8.29% |

#### 预填充阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 344 | 27533.915 | 80.040 | 100.00% |
| `decoder.attn_hc_pre` | 344 | 200.418 | 0.583 | 0.73% |
| `decoder.attn_norm` | 344 | 146.549 | 0.426 | 0.53% |
| `decoder.attn_hc_post` | 344 | 68.592 | 0.199 | 0.25% |
| `decoder.ffn_hc_pre` | 344 | 274.393 | 0.798 | 1.00% |
| `decoder.ffn_norm` | 344 | 207.701 | 0.604 | 0.75% |
| `decoder.moe` | 344 | 26578.679 | 77.264 | 96.53% |
| `decoder.ffn_hc_post` | 344 | 26.041 | 0.076 | 0.09% |
| `moe.forward` | 344 | 26574.015 | 77.250 | 96.51% |
| `moe.topk` | 344 | 20868.684 | 60.665 | 75.79% |
| `moe.experts` | 344 | 2693.175 | 7.829 | 9.78% |
| `moe.shared_experts` | 344 | 1554.629 | 4.519 | 5.65% |
| `moe.add_shared` | 344 | 189.569 | 0.551 | 0.69% |
| `moe.allreduce` | 344 | 1251.206 | 3.637 | 4.54% |

#### 解码阶段

| 事件 | 调用次数 | 总耗时 ms | 平均耗时 ms | 占 decoder.layer 比例 |
| --- | ---: | ---: | ---: | ---: |
| `decoder.layer` | 5160 | 26304.063 | 5.098 | 100.00% |
| `decoder.attn_hc_pre` | 5160 | 844.329 | 0.164 | 3.21% |
| `decoder.attn_norm` | 5160 | 249.710 | 0.048 | 0.95% |
| `decoder.attn_hc_post` | 5160 | 287.521 | 0.056 | 1.09% |
| `decoder.ffn_hc_pre` | 5160 | 773.360 | 0.150 | 2.94% |
| `decoder.ffn_norm` | 5160 | 236.680 | 0.046 | 0.90% |
| `decoder.moe` | 5160 | 23147.913 | 4.486 | 88.00% |
| `decoder.ffn_hc_post` | 5160 | 336.564 | 0.065 | 1.28% |
| `moe.forward` | 5160 | 23081.252 | 4.473 | 87.75% |
| `moe.topk` | 5160 | 1942.429 | 0.376 | 7.38% |
| `moe.experts` | 5160 | 16037.694 | 3.108 | 60.97% |
| `moe.shared_experts` | 5160 | 1328.912 | 0.258 | 5.05% |
| `moe.add_shared` | 5160 | 320.047 | 0.062 | 1.22% |
| `moe.allreduce` | 5160 | 3209.785 | 0.622 | 12.20% |
