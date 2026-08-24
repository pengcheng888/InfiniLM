# Qwen3MoeSparseMoeBlock 非 legacy MoE forward 调用链分析

本文分析 `/workspace_codex/InfiniLM/csrc/models/qwen3_moe/qwen3_moe_sparse_moe_block.cpp` 中
`Qwen3MoeSparseMoeBlock::forward` 在 `use_legacy_moe_ == false` 时的函数调用和算子调用过程。

## 入口条件

`Qwen3MoeSparseMoeBlock` 构造时会根据配置决定是否走 legacy 分支：

```cpp
use_legacy_moe_ =
    model_config->get_or<std::string>("model_type", "") == "qwen3_moe" &&
    model_config->get_or<bool>("use_legacy_moe", false);
```

当 `use_legacy_moe_ == false` 时，构造函数注册的是通用 MoE 组件：

- `gate_`: `infinilm::layers::moe::TopKRouter`
- `experts_`: `infinilm::layers::moe::FusedMoeExperts`
- `fused_moe_`: `infinilm::layers::moe::FusedMoE`

对应源码位置：`csrc/models/qwen3_moe/qwen3_moe_sparse_moe_block.cpp:15-24`。

## 总体 forward 流程

`forward` 输入要求是 3D tensor，形状通常为：

```text
[batch, seq_len, hidden_size]
```

非 legacy 分支的过程是：

1. 将输入 reshape/view 成 2D：

```cpp
hidden_states_reshaped = hidden_states->view({batch * seq_len, hidden_size});
```

2. 调用通用 router：

```cpp
auto [routing_weights, selected_experts] = gate_->forward(hidden_states_reshaped);
```

3. 将 router 输出包装成 `TopKOutput`：

```cpp
TopKOutput topk_output{
    routing_weights,
    selected_experts,
    infinicore::Tensor(),
};
```

这里 `router_logits` 为空 tensor，后续 fused MoE 路径只使用 `topk_weights` 和 `topk_ids`。

4. 调用通用 fused MoE：

```cpp
auto final_hidden_states = fused_moe_->forward(
    hidden_states_reshaped,
    topk_output,
    experts_->moe_weights());
```

5. 将输出 view 回原始 3D 形状：

```cpp
return final_hidden_states->view({batch, seq_len, hidden_size});
```

对应源码位置：`csrc/models/qwen3_moe/qwen3_moe_sparse_moe_block.cpp:27-51`。

简化调用链如下：

```text
Qwen3MoeSparseMoeBlock::forward
  -> TopKRouter::forward
       -> infinicore::op::linear
       -> infinicore::op::moe_topk_softmax_
          或 infinicore::op::moe_topk_sigmoid_
          或 infinicore::op::moe_fused_gate_
  -> FusedMoeExperts::moe_weights
  -> FusedMoE::forward
       -> dispatcher_->dispatch
       -> CudaFusedMoeRunner::run
            -> prepare_runner_input
                 -> infinicore::op::moe_align_
                    或 infinicore::op::moe_align_with_expert_map_
            -> run_fused_core
                 -> infinicore::op::moe_fused_dense_
       -> dispatcher_->combine
            -> 可选 infinicore::op::distributed::allreduce_
```

## Router 阶段

`TopKRouter::forward` 输入是已经 flatten 后的 2D tensor：

```text
[num_tokens, hidden_size]
```

它首先调用：

```cpp
auto router_logits = infinicore::op::linear(hidden_states, weight_, std::nullopt, 1.0f);
```

其中 `weight_` 形状为：

```text
[num_experts, hidden_size]
```

`infinicore::op::linear` 内部会创建输出，并调用 `linear_`；`linear_` 将输入 view 为 `[N, in_features]`，将权重 `permute({1, 0})` 后调用 `gemm_`。也就是说 router logits 的实际矩阵乘是：

```text
[num_tokens, hidden_size] x [hidden_size, num_experts]
```

得到：

```text
router_logits: [num_tokens, num_experts]
```

随后 router 分配两个输出 tensor：

- `router_scores`: `[num_tokens, num_experts_per_tok]`, `F32`
- `router_indices`: `[num_tokens, num_experts_per_tok]`, `I32`

然后根据 `router_backend_` 调用不同 InfiniCore MoE topk 算子：

- `TopKRouterBackend::Softmax`
  - 调用 `infinicore::op::moe_topk_softmax_`
  - 输入 `router_logits`，可选 `e_score_correction_bias_`
  - 参数包括 `norm_topk_prob_` 和 `moe_softcapping_`

- `TopKRouterBackend::Sigmoid`
  - 调用 `infinicore::op::moe_topk_sigmoid_`
  - 输入 `router_logits`，可选 `e_score_correction_bias_`
  - 参数包括 `norm_topk_prob_`

- `TopKRouterBackend::FusedGate`
  - 调用 `infinicore::op::moe_fused_gate_`
  - 输入 `router_logits` 和 `e_score_correction_bias_`
  - 参数包括 `num_expert_group_`、`topk_group_`、`num_fused_shared_experts_`、`routed_scaling_factor_`、`apply_routed_scaling_factor_on_output_`

对应源码位置：

- `csrc/layers/moe/router/topk_router.cpp:41-78`: router 配置和参数初始化
- `csrc/layers/moe/router/topk_router.cpp:80-123`: router forward 和 topk 算子调用
- `/workspace_codex/InfiniCore/src/infinicore/ops/linear/linear.cc`: `linear` 到 `gemm_` 的实现

## 专家权重布局

非 legacy 分支不再通过 `Qwen3MoeExperts` 逐专家调用旧的 `fused_moe` 接口，而是使用 `FusedMoeExperts` 在加载期准备 packed 权重。

`FusedMoeExperts` 会注册两个 packed 参数：

```text
w13_weight_: [num_local_experts, intermediate_size_per_partition * 2, hidden_size]
w2_weight_:  [num_local_experts, hidden_size, intermediate_size_per_partition]
```

其中 `w13_weight_` 的第 1 维前半段对应 `gate_proj.weight`，后半段对应 `up_proj.weight`；`w2_weight_` 对应 `down_proj.weight`。

为了兼容原始权重命名，它还给每个本地 expert 注册 view 参数：

- `<global_expert>.gate_proj.weight`
- `<global_expert>.up_proj.weight`
- `<global_expert>.down_proj.weight`

最终 `experts_->moe_weights()` 返回：

```cpp
MoeWeights{
    packed_w13 = w13_weight_,
    packed_w2 = w2_weight_,
}
```

对应源码位置：

- `csrc/layers/moe/experts/fused_moe_experts.cpp:10-44`: packed 参数创建
- `csrc/layers/moe/experts/fused_moe_experts.cpp:46-68`: 每个 expert 的 view 参数注册
- `csrc/layers/moe/experts/fused_moe_experts.cpp:70-75`: `MoeWeights` 返回

## FusedMoE 阶段

`FusedMoE::forward` 做三件事：

```cpp
auto dispatch_output = dispatcher_->dispatch(hidden_states, topk_output, workspace_);
auto combine_input = runner_->run(dispatch_output, weights, workspace_);
return dispatcher_->combine(combine_input, workspace_);
```

对应源码位置：`csrc/layers/moe/fused_moe.cpp:41-46`。

构造 `FusedMoE` 时会根据 EP 配置创建 dispatcher，并固定创建 `CudaFusedMoeRunner`：

- `EPBackend::Disabled`: `StandardDispatcher`
- `EPBackend::AllGatherReduceScatter`: `AllGatherReduceScatterDispatcher`
- `EPBackend::LocalAllReduce`: `LocalAllReduceDispatcher`
- `EPBackend::DeepEP`: `DeepEPDispatcher`

对应源码位置：

- `csrc/layers/moe/fused_moe.cpp:18-38`
- `csrc/layers/moe/dispatcher/dispatcher_factory.cpp:12-26`

## StandardDispatcher 路径

默认 EP disabled 时使用 `StandardDispatcher`。

`dispatch` 不移动 token，也不创建 expert map，只是把原始 `hidden_states` 和 `topk_output` 打包成 `DispatchOutput`：

```cpp
return DispatchOutput{
    DispatchOutputFormat::Standard,
    hidden_states,
    infinicore::Tensor(),
    topk_output,
};
```

`combine` 在 TP world size 大于 1 且 communicator 存在时调用：

```cpp
infinicore::op::distributed::allreduce_(
    combine_input.hidden_states,
    combine_input.hidden_states,
    INFINICCL_SUM,
    communicator_);
```

否则直接返回 runner 输出。

对应源码位置：`csrc/layers/moe/dispatcher/standard_dispatcher.cpp:11-39`。

## Runner 阶段

`CudaFusedMoeRunner::run` 分两步：

```cpp
auto runner_input = prepare_runner_input(dispatch_output, workspace);
auto runner_output = run_fused_core(runner_input, weights, workspace);
```

对应源码位置：`csrc/layers/moe/runner/cuda_fused_moe_runner.cpp:76-90`。

### prepare_runner_input

`prepare_runner_input` 读取 `topk_ids`，要求其为 2D tensor：

```text
[num_tokens, num_experts_per_tok]
```

它根据 `num_pairs = num_tokens * num_experts_per_tok`、`num_local_experts_` 和 `align_block_size_` 计算 workspace 容量，确保以下 workspace tensor 已分配：

- `workspace.sorted_token_ids`: `I32`
- `workspace.expert_ids`: `I32`
- `workspace.num_tokens_post_padded`: `I32[1]`

graph recording 期间如果 workspace 未提前初始化，会抛错：

```cpp
if (infinicore::context::isGraphRecording()) {
    throw std::runtime_error("MoE ... workspace was not initialized before graph capture");
}
```

然后根据 `dispatch_output.expert_map` 是否存在调用不同 align 算子：

- 无 `expert_map`：

```cpp
infinicore::op::moe_align_(
    workspace.sorted_token_ids,
    workspace.expert_ids,
    workspace.num_tokens_post_padded,
    topk_ids,
    num_local_experts_,
    block_size,
    true);
```

- 有 `expert_map`：

```cpp
infinicore::op::moe_align_with_expert_map_(
    workspace.sorted_token_ids,
    workspace.expert_ids,
    workspace.num_tokens_post_padded,
    topk_ids,
    dispatch_output.expert_map,
    num_local_experts_,
    block_size,
    true);
```

`moe_align_` 的作用是把 topk 路由结果整理成 fused expert kernel 需要的分块元数据：

- `sorted_token_ids`
- `expert_ids`
- `num_tokens_post_padded`

对应源码位置：`csrc/layers/moe/runner/cuda_fused_moe_runner.cpp:93-163`。

### run_fused_core

`run_fused_core` 首先检查 packed 权重存在且满足：

```text
w13: [num_local_experts, intermediate_size_per_partition * 2, hidden_size]
w2:  [num_local_experts, hidden_size, intermediate_size_per_partition]
```

并要求权重和输入在同一 device、同一 dtype。

然后确保 `workspace.fused_moe_output` 的形状、dtype、device 与输入 hidden states 一致：

```text
workspace.fused_moe_output: [num_tokens, hidden_size]
```

最后调用：

```cpp
infinicore::op::moe_fused_dense_(
    workspace.fused_moe_output,
    runner_input.hidden_states,
    weights.packed_w13,
    weights.packed_w2,
    runner_input.topk_output.topk_weights,
    runner_input.topk_output.topk_ids,
    runner_input.routing_metadata.sorted_token_ids,
    runner_input.routing_metadata.expert_ids,
    runner_input.routing_metadata.num_tokens_post_padded);
```

这个算子完成 routed experts 的核心计算，可抽象为每个 token 对其 top-k experts 执行：

```text
gate_up = hidden @ w13[expert].T
gate = gate_up[:, :intermediate]
up = gate_up[:, intermediate:]
activated = silu(gate) * up
down = activated @ w2[expert].T
output[token] += down * topk_weight
```

对应源码位置：`csrc/layers/moe/runner/cuda_fused_moe_runner.cpp:165-201`。

## InfiniCore 算子入口和后端分派

本路径涉及的主要 InfiniCore 算子如下。

| 阶段 | InfiniLM 调用 | InfiniCore C++ API | 主要作用 |
| --- | --- | --- | --- |
| router linear | `infinicore::op::linear` | `include/infinicore/ops/linear.hpp` | 计算 router logits |
| router topk softmax | `moe_topk_softmax_` | `include/infinicore/ops/moe_topk_softmax.hpp` | softmax 打分并选 top-k expert |
| router topk sigmoid | `moe_topk_sigmoid_` | `include/infinicore/ops/moe_topk_sigmoid.hpp` | sigmoid 打分并选 top-k expert |
| fused gate | `moe_fused_gate_` | `include/infinicore/ops/moe_fused_gate.hpp` | noaux/fused gate 路由 |
| align | `moe_align_` | `include/infinicore/ops/moe_align.hpp` | 生成 fused MoE 的 token/expert 分块元数据 |
| align with map | `moe_align_with_expert_map_` | `include/infinicore/ops/moe_align.hpp` | 带 expert map 的 align |
| fused experts | `moe_fused_dense_` | `include/infinicore/ops/moe_fused_dense.hpp` | 执行 packed w13/w2 expert 前向并按 topk 权重累加 |
| TP combine | `distributed::allreduce_` | `infinicore/ops/distributed/allreduce.hpp` | TP 多卡时对局部 expert 输出求和 |

`moe_topk_softmax_`、`moe_topk_sigmoid_`、`moe_fused_gate_`、`moe_align_`、`moe_fused_dense_` 都是 graph-aware op 包装，常规模式下直接运行设备实现，graph recording 时通过 `INFINICORE_GRAPH_OP_RECORD_OR_RUN` 记录或执行。

需要注意当前 Hygon ATen 特化：

- `moe_align_` 在 `ENABLE_ATEN && ENABLE_HYGON_API` 且输入 device 为 HYGON 时，会直接调用 `deepseek_v4_lightop_moe_align_block_size_`，不走通用 `MoeAlign::execute`。
- `moe_align_with_expert_map_` 在同样条件下会直接调用 `deepseek_v4_lightop_moe_align_block_size_with_expert_map_`。
- `moe_fused_dense_` 在 `ENABLE_ATEN && ENABLE_HYGON_API` 且输出 device 为 HYGON 时，会直接调用当前文件内的 `moe_fused_dense_aten_` naive 实现，忽略 `sorted_token_ids`、`expert_ids`、`num_tokens_post_padded`。

对应 InfiniCore 源码位置：

- `/workspace_codex/InfiniCore/src/infinicore/ops/moe_topk_softmax/moe_topk_softmax.cc`
- `/workspace_codex/InfiniCore/src/infinicore/ops/moe_topk_sigmoid/moe_topk_sigmoid.cc`
- `/workspace_codex/InfiniCore/src/infinicore/ops/moe_fused_gate/moe_fused_gate.cc`
- `/workspace_codex/InfiniCore/src/infinicore/ops/moe_align/moe_align.cc`
- `/workspace_codex/InfiniCore/src/infinicore/ops/moe_fused_dense/moe_fused_dense.cc`

## 数据流总结

以 `EPBackend::Disabled` 的默认路径为例，数据流是：

```text
hidden_states [B, S, H]
  -> view
hidden_states_reshaped [B*S, H]
  -> TopKRouter::forward
router_logits [B*S, E]
  -> moe_topk_* / moe_fused_gate_
topk_weights [B*S, K], topk_ids [B*S, K]
  -> TopKOutput
  -> StandardDispatcher::dispatch
DispatchOutput(hidden_states_reshaped, topk_output)
  -> CudaFusedMoeRunner::prepare_runner_input
sorted_token_ids / expert_ids / num_tokens_post_padded
  -> CudaFusedMoeRunner::run_fused_core
fused_moe_output [B*S, H]
  -> StandardDispatcher::combine
  -> view
final_hidden_states [B, S, H]
```

## 和 legacy 分支的关键区别

legacy 分支使用 Qwen3 专用的：

- `Qwen3MoeTopKRouter`
- `Qwen3MoeExperts`

其中旧专家路径会调用 `infinicore::op::fused_moe(...)` 这类较老接口。

非 legacy 分支则改为：

- 通用 `TopKRouter`
- 通用 `FusedMoeExperts`
- 通用 `FusedMoE`
- packed `w13/w2` 权重布局
- `moe_align_` + `moe_fused_dense_` 的新 fused runner 路径

因此，非 legacy 分支的模型层代码很薄，主要负责 reshape、router 输出包装、调用通用 fused MoE，核心 MoE 计算集中在 `csrc/layers/moe` 和 InfiniCore 的 MoE op 中。
