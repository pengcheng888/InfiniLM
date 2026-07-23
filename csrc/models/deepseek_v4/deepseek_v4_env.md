# DeepSeek V4 环境变量说明

本文档汇总 InfiniLM DeepSeek V4 路径使用的专用环境变量。当前约定是：这些环境变量在进程启动后保持不变，模型代码在构造阶段或首次静态初始化时读取，避免在每次 `forward` 中反复访问环境变量。

## MHC 后端选择

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINILM_DSV4_MHC_PRE` | `naive` | `naive`, `kernel` | `DeepseekV4DecoderLayer` 构造函数 | 选择 MHC pre 算子，同时作用于 attention 侧和 FFN 侧的 MHC pre。 |
| `INFINILM_DSV4_MHC_POST` | `naive` | `naive`, `kernel` | `DeepseekV4DecoderLayer` 构造函数 | 选择 MHC post 算子，同时作用于 attention 侧和 FFN 侧的 MHC post。 |
| `INFINILM_DSV4_MHC_HEAD` | `naive` | `naive`, `kernel` | `DeepseekV4Model` 构造函数 | 选择 final norm 前的 MHC head collapse 算子。 |

MHC kernel 选择接受的真值别名包括 `1`、`true`、`TRUE`、`on`、`ON`、`kernel`。naive/假值别名包括 `0`、`false`、`FALSE`、`off`、`OFF`、`naive`。其它取值会报错。

## Gate/TopK 后端选择

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINILM_DSV4_GATE_TOPK` | `naive` | `naive`, `kernel` | `DeepseekV4MoEGate` 构造函数 | 选择 gate 之后 routed expert topk 的计算路径。hash MoE 层对应 `deepseek_v4_hash_topk_*`，非 hash MoE 层对应带 correction bias 的 `deepseek_v4_topk_*`。 |

Gate/TopK kernel 选择接受的真值/假值别名与 MHC kernel 一致。默认保持 `naive`，用于保证现有精度行为不变；设置为 `kernel` 后使用 InfiniCore native 单 kernel 实现以减少 ATen 多算子和中间张量开销。

## 路由专家后端

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINILM_DSV4_ROUTED_EXPERT_BACKEND` | `naive` | `naive`, `lmslim_fused`, `fused_experts_int8_marlin`, `aiter_split`, `lightop_split` | `DeepseekV4PackedExperts` 构造函数 | 选择 routed expert 的计算路径。 |
| `INFINILM_DSV4_FUSED_SHARED_OUTPUT` | 关闭 | `1`, `true`, `TRUE`, `on`, `ON` | `DeepseekV4MoE` 构造函数 | 实验性开关。仅在 `fused_experts_int8_marlin` 后端下，将 shared experts 输出传入 InfiniCore fused expert 算子内部做 `routed * scaling + shared`，用于评估去掉外部 `moe.add_shared` 的收益。默认关闭以保持当前 token 一致性。 |

兼容别名：`reference` 等价于 `naive`，`lmslim` 和 `fused` 等价于 `lmslim_fused`，`int8_marlin` 和 `sglang_int8_marlin` 等价于 `fused_experts_int8_marlin`，`aiter` 等价于 `aiter_split`，`lightop` 和 `split_lightop` 等价于 `lightop_split`。

后端行为：

| 后端 | 说明 |
| --- | --- |
| `naive` | 调用 InfiniCore 的 naive W8A8 MoE 参考算子。 |
| `lmslim_fused` | 保留的旧 fused lmslim/Marlin 后端名，调用旧 InfiniCore 入口，内部同样走 SGLang `fused_experts_impl_int8_marlin`。 |
| `fused_experts_int8_marlin` | 新增的显式 SGLang fused expert 后端名，调用 InfiniCore `deepseek_v4_fused_experts_impl_int8_marlin_`，最终走 `sglang::fused_experts_impl_int8_marlin`。 |
| `aiter_split` | 运行 AITER 风格的拆分 Marlin 路径：量化、对齐、GEMM、激活、GEMM、求和。 |
| `lightop_split` | 运行 lightop 风格的拆分 Marlin 路径。 |

注意：非 `naive` 后端会在 `process_weights_after_loading()` 中生成 Marlin 格式权重。完整权重场景下，为降低常驻显存，当前实现会在 repack 完成后释放原始 `w13_weight`/`w2_weight` 成员引用，并禁止再回退到 `naive`。

## Marlin GEMM 调参覆盖

以下变量不负责选择 routed expert 后端，只在已选择的后端使用 Marlin 时覆盖内部 GEMM 调参。它们会在 `DeepseekV4PackedExperts` 构造函数中读取到 `marlin_gemm_override_`。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `INFINILM_DSV4_MOE_MARLIN_BLOCK_SIZE` | 自动 | 覆盖传给 MoE token alignment 和 Marlin GEMM 配置的 block size。 |
| `INFINILM_DSV4_MOE_MARLIN_MODE` | 自动 | 同时覆盖 GEMM1 和 GEMM2 的 mode。 |
| `INFINILM_DSV4_MOE_MARLIN_MODE1` | 自动 | 只覆盖 GEMM1 的 mode。 |
| `INFINILM_DSV4_MOE_MARLIN_MODE2` | 自动 | 只覆盖 GEMM2 的 mode。 |
| `INFINILM_DSV4_MOE_MARLIN_DELTA` | 自动 | 覆盖 Marlin delta 参数。 |

## 调试与性能分析

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINILM_DSV4_DEBUG_DUMP` | 关闭 | `1`, `true`, `TRUE`, `on`, `ON` | `DeepseekV4DecoderLayer` 和 `DeepseekV4MoE` 构造函数 | 将部分中间 tensor dump 到 `/tmp/infinilm_dsv4_tp*_l*_*`。 |
| `INFINILM_DSV4_PROFILE` | 关闭 | `1`, `true`, `TRUE`, `on`, `ON` | `deepseek_v4_profile.hpp` 静态初始化 | 开启 DeepSeek V4 性能分析计时并在退出时打印报告。 |
| `INFINILM_DSV4_FFN_PROFILE` | 关闭 | `1`, `true`, `TRUE`, `on`, `ON` | `deepseek_v4_profile.hpp` 静态初始化 | 保留的 FFN 性能分析别名开关。 |

注意：当前性能分析计时器会在每个计时代码块前后同步 GPU stream，因此适合定位热点，不适合作为关闭 profile 后的真实性能数据。

## 已废弃的后端别名

路由专家后端选择已经统一集中到 `INFINILM_DSV4_ROUTED_EXPERT_BACKEND`。以下早期后端选择别名不应继续在模型代码中使用：

- `INFINILM_DSV4_MOE_MARLIN`
- `INFINILM_DSV4_MOE_BACKEND`
- `INFINILM_DSV4_MOE_AITER`
- `INFINILM_DSV4_MOE_SPLIT_LIGHTOP`
