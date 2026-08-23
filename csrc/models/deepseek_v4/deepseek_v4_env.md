# DeepSeek V4 环境变量说明

本文档汇总 InfiniLM DeepSeek V4 路径当前源码实际读取的专用环境变量。除特别说明外，建议在进程启动前设置这些变量，并在同一进程内保持不变。

注意：本文以当前源码为准。部分历史变量仍出现在旧 profile 报告或脚本中，但模型路径已经不再读取。

## MHC 算子

当前 InfiniLM DeepSeek V4 模型路径固定调用 InfiniCore kernel 版本：

| 阶段 | 固定算子 |
| --- | --- |
| MHC pre | `deepseek_v4_mhc_pre_kernel_` |
| MHC post | `deepseek_v4_mhc_post_kernel_` |
| MHC head | `deepseek_v4_hc_head_kernel_` |

MHC 后端选择环境变量已删除；模型 forward 不再读取环境变量，也不再支持切换到 naive 版本。`deepseek_v4_mhc_pre_naive_`、`deepseek_v4_mhc_post_naive_`、`deepseek_v4_hc_head_naive_` 仅作为 InfiniCore 层历史/测试接口保留，DeepSeek V4 模型路径不再考虑这三个算子。

## Gate/TopK

当前 `DeepseekV4MoEGate::forward` 固定调用 InfiniCore public kernel 入口：

| 场景 | 固定算子 |
| --- | --- |
| hash MoE 层 | `deepseek_v4_hash_topk_` |
| 非 hash MoE 层 | `deepseek_v4_topk_` |

`INFINILM_DSV4_GATE_TOPK` 已不再被当前模型源码读取。模型侧不再提供 naive/kernel 环境变量切换；如需排查 topk 行为，应在 InfiniCore public op 内部或测试路径中处理。

## 路由专家后端

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINILM_DSV4_ROUTED_EXPERT_BACKEND` | `fused_experts_int8_marlin` | `naive`, `fused_experts_int8_marlin`, `aiter_split`, `lightop_split` | `DeepseekV4PackedExperts` 构造函数 | 选择 routed expert 的计算路径。 |

兼容别名：`reference` 等价于 `naive`，`int8_marlin` 和 `sglang_int8_marlin` 等价于 `fused_experts_int8_marlin`，`aiter` 等价于 `aiter_split`，`lightop` 和 `split_lightop` 等价于 `lightop_split`。

当前 shared output 融合固定启用：`DeepseekV4MoE::forward` 会把 shared experts 输出作为可选参数传给 routed expert backend。

后端行为：

| 后端 | 说明 |
| --- | --- |
| `naive` | 调用 InfiniCore 的 naive W8A8 MoE 参考算子。 |
| `fused_experts_int8_marlin` | 显式 INT8 Marlin fused expert 后端名，调用 InfiniCore native `deepseek_v4_fused_experts_impl_int8_marlin_`。 |
| `aiter_split` | 运行 AITER 风格的拆分 Marlin 路径：量化、对齐、GEMM、激活、GEMM、求和。 |
| `lightop_split` | 运行 lightop 风格的拆分 Marlin 路径。 |

注意：非 `naive` 后端会在 `process_weights_after_loading()` 中生成 Marlin 格式权重。完整权重场景下，为降低常驻显存，当前实现会在 repack 完成后释放原始 `w13_weight`/`w2_weight` 成员引用，并禁止再回退到 `naive`。


## MoE AllReduce 后端选择

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINILM_DSV4_MOE_ALLREDUCE` | `inplace` | `inplace`, `outplace`, `custom`, `dcu_custom`, `custom_ar` | `DeepseekV4MoE` 构造函数 | 选择 MoE TP allreduce 路径。`inplace` 为原始 `infinicclAllReduce(routed, routed)`；`outplace` 使用 per-layer scratch buffer 做 out-of-place `infinicclAllReduce`；`custom`/`dcu_custom`/`custom_ar` 会先尝试 InfiniCore DCU custom allreduce wrapper，失败时回退到 `infiniccl`。 |

当前实测 `outplace` 在 Hygon layer0-4 case 中没有收益，默认保持 `inplace`。`custom` 用于验证对齐 SGLang/vLLM `_C_custom_ar` 的可行性；由于 InfiniLM 是单进程多 rank 线程模型，而 SGLang 是多进程模型，vLLM IPC handle 打开路径在当前进程模型下仍可能失败并回退。

InfiniCore distributed allreduce 还支持一个更底层的 DeepSeek V4/Hygon fast path 开关：

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINICORE_ALLREDUCE_FASTPATH` | `off` | `off`, `0`, `false`, `FALSE`, `deepseek_v4`, `dsv4`, `hygon_deepseek_v4`, `1`, `true`, `TRUE`, `on`, `ON` | InfiniCore `distributed::AllReduce::run` | 在通用 `distributed::allreduce_` 内部优先尝试 `deepseek_v4_dcu_custom_allreduce_`，成功则跳过 `infinicclAllReduce`，失败自动回退。该 fast path 只对 `INFINICCL_SUM`、有效 communicator 生效。 |

## InfiniCore FlashMLA 兼容开关

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINICORE_DSV4_FLASHMLA_FORCE_NAIVE` | 关闭 | `1`, `true`, `TRUE`, `on`, `ON` | InfiniCore `deepseek_v4_flashmla_sparse_attention_*` 首次调用 | 强制 DeepSeek V4 FlashMLA sparse attention 不加载 `flash_mla` SO，改走 ATen naive/reference 计算。用于目标容器没有可用 `flash_mla` wheel/SO、或 FlashMLA 符号不兼容时验证服务可用性；该路径性能很慢，不应用作性能测试口径。开启后 `with_metadata` 返回空 schedule，带 graph 的 decode 不会复用 FlashMLA metadata 路径。 |

测试注意：开启该变量后，FlashMLA sparse attention 的测试应使用容差比较或可用性 smoke test，不应要求与 FlashMLA kernel `torch.equal` 位级一致。

## InfiniLM FlashMLA schedule 预分配

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINILM_DSV4_FLASHMLA_PREALLOC_METADATA` | 开启 | 关闭值：`0`, `false`, `FALSE`, `off`, `OFF`, `no`, `NO` | DeepSeek V4 graph metadata 绑定 | 在 InfiniLM 中提前分配 FlashMLA schedule tensor：`tile_scheduler_metadata=[160, 8]`、`num_splits=[tokens + 1]`。默认开启后 schedule 空间不再依赖首次 `attn_->forward`/`with_metadata_` 创建，graph capture 仍会通过 `deepseek_v4_flashmla_sparse_attention_metadata_` 刷新内容；显式关闭可回到旧路径。 |

## InfiniCore FlashMLA 兼容开关

| 环境变量 | 默认值 | 可选值 | 读取位置 | 作用 |
| --- | --- | --- | --- | --- |
| `INFINICORE_DSV4_FLASHMLA_FORCE_NAIVE` | 关闭 | `1`, `true`, `TRUE`, `on`, `ON` | InfiniCore `deepseek_v4_flashmla_sparse_attention_*` 首次调用 | 强制 DeepSeek V4 FlashMLA sparse attention 不加载 `flash_mla` SO，改走 ATen naive/reference 计算。用于目标容器没有可用 `flash_mla` wheel/SO、或 FlashMLA 符号不兼容时验证服务可用性；该路径性能很慢，不应用作性能测试口径。开启后 `with_metadata` 返回空 schedule，带 graph 的 decode 不会复用 FlashMLA metadata 路径。 |

测试注意：开启该变量后，FlashMLA sparse attention 的测试应使用容差比较或可用性 smoke test，不应要求与 FlashMLA kernel `torch.equal` 位级一致。


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
| `INFINILM_DSV4_DEBUG_DUMP` | 关闭 | `1`, `true`, `TRUE`, `on`, `ON`, `kernel`, `marlin` | `DeepseekV4DecoderLayer` 和 `DeepseekV4MoE` 构造函数 | 将部分中间 tensor dump 到 `/tmp/infinilm_dsv4_tp*_l*_*`。 |
| `INFINILM_DSV4_PROFILE` | 关闭 | 任意非空且首字符不是 `0` 的值 | `deepseek_v4_profile.hpp` 静态初始化 | 开启 DeepSeek V4 GPU-synced wall time 计时，并在进程退出时打印 overall/prefill/decode 报告。 |
| `INFINILM_DSV4_FFN_PROFILE` | 关闭 | 任意非空且首字符不是 `0` 的值 | `deepseek_v4_profile.hpp` 静态初始化 | 保留的 FFN 性能分析别名开关；当前与 `INFINILM_DSV4_PROFILE` 一样会开启完整 DSv4 profile。 |
| `INFINILM_DSV4_PROCESSOR_PROFILE` | 关闭 | `1`, `true`, `yes`, `on` | `DeepSeekV4Processor` 每次构造模型输入时 | 打印 Python processor 侧耗时，包括基础输入构造、DSv4 attention metadata 构造、list 生成和 `infinicore.from_list` 耗时。 |

注意：当前性能分析计时器会在每个计时代码块前后同步 GPU stream，因此适合定位热点，不适合作为关闭 profile 后的真实性能数据。

## 已废弃的后端别名

路由专家后端选择已经统一集中到 `INFINILM_DSV4_ROUTED_EXPERT_BACKEND`。以下早期后端选择别名不应继续在模型代码中使用：

- `INFINILM_DSV4_GATE_TOPK`
- `INFINILM_DSV4_MOE_MARLIN`
- `INFINILM_DSV4_MOE_BACKEND`
- `INFINILM_DSV4_MOE_AITER`
- `INFINILM_DSV4_MOE_SPLIT_LIGHTOP`
