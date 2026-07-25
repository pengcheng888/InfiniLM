# DeepSeek V4 环境变量说明

本文档汇总 InfiniLM DeepSeek V4 路径使用的专用环境变量。当前约定是：这些环境变量在进程启动后保持不变，模型代码在构造阶段或首次静态初始化时读取，避免在每次 `forward` 中反复访问环境变量。

## MHC 算子

当前 InfiniLM DeepSeek V4 模型路径固定调用 InfiniCore kernel 版本：

| 阶段 | 固定算子 |
| --- | --- |
| MHC pre | `deepseek_v4_mhc_pre_kernel_` |
| MHC post | `deepseek_v4_mhc_post_kernel_` |
| MHC head | `deepseek_v4_mhc_head_kernel_` |

MHC 后端选择环境变量已删除；模型 forward 不再读取环境变量，也不再支持切换到 naive 版本。`deepseek_v4_mhc_pre_naive_`、`deepseek_v4_mhc_post_naive_`、`deepseek_v4_mhc_head_naive_` 仅作为 InfiniCore 层历史/测试接口保留，DeepSeek V4 模型路径不再考虑这三个算子。

## Gate/TopK 后端选择

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINILM_DSV4_GATE_TOPK` | `naive` | `naive`, `kernel` | `DeepseekV4MoEGate` 构造函数 | 选择 gate 之后 routed expert topk 的计算路径。hash MoE 层对应 `deepseek_v4_hash_topk_*`，非 hash MoE 层对应带 correction bias 的 `deepseek_v4_topk_*`。 |

Gate/TopK kernel 选择接受常见真值/假值别名。默认保持 `naive`，用于保证现有精度行为不变；设置为 `kernel` 后使用 InfiniCore native 单 kernel 实现以减少 ATen 多算子和中间张量开销。

## 路由专家后端

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINILM_DSV4_ROUTED_EXPERT_BACKEND` | `naive` | `naive`, `lmslim_fused`, `fused_experts_int8_marlin`, `aiter_split`, `lightop_split` | `DeepseekV4PackedExperts` 构造函数 | 选择 routed expert 的计算路径。 |
| `INFINILM_DSV4_FUSED_SHARED_OUTPUT` | `auto` | `auto`, `1`, `true`, `TRUE`, `on`, `ON`, `0`, `false`, `FALSE`, `off`, `OFF` | `DeepseekV4MoE` 构造函数 | 控制 shared experts 输出是否融合进 routed expert 后处理。默认 `auto`：仅在 `fused_experts_int8_marlin` 支持该路径时启用，将 `routed * scaling + shared` 放入 InfiniCore fused expert 算子内部，避免外部 `infinicore::op::add`。 |

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


## MoE AllReduce 后端选择

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINILM_DSV4_MOE_ALLREDUCE` | `inplace` | `inplace`, `outplace`, `custom` | `DeepseekV4MoE` 构造函数 | 选择 MoE TP allreduce 路径。`inplace` 为原始 `infinicclAllReduce(routed, routed)`；`outplace` 使用 per-layer scratch buffer 做 out-of-place `infinicclAllReduce`；`custom` 尝试调用 InfiniCore DCU custom allreduce wrapper，失败时回退到 `infiniccl`。 |

当前实测 `outplace` 在 Hygon layer0-4 case 中没有收益，默认保持 `inplace`。`custom` 用于验证对齐 SGLang/vLLM `_C_custom_ar` 的可行性；由于 InfiniLM 是单进程多 rank 线程模型，而 SGLang 是多进程模型，vLLM IPC handle 打开路径在当前进程模型下仍可能失败并回退。

InfiniCore distributed allreduce 还支持一个更底层的 DeepSeek V4/Hygon fast path 开关：

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINICORE_ALLREDUCE_FASTPATH` | `off` | `off`, `deepseek_v4`, `dsv4`, `hygon_deepseek_v4`, `1`, `true`, `on` | InfiniCore `distributed::AllReduce::run` | 在通用 `distributed::allreduce_` 内部优先尝试 `deepseek_v4_dcu_custom_allreduce_`，成功则跳过 `infinicclAllReduce`，失败自动回退。该 fast path 由 DeepSeek V4/Hygon 专用 wrapper 负责检查 tensor 连续性、大小、对齐和 world size。 |


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
