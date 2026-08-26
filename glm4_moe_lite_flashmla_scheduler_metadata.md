# GLM4 MoE Lite FlashMLA Scheduler Metadata 使用过程

本文整理 `glm4_moe_lite` 模型中 `FlashMLAMetadata::scheduler_metadata` 的来源、生命周期和在 eager / graph 路径中的使用方式。

## 相关结构

`FlashMLAMetadata` 定义在 `csrc/layers/mla_attention/backends/flashmla.hpp`，包含四类数据：

```cpp
struct FlashMLAMetadata {
    infinicore::Tensor slot_mapping;
    infinicore::Tensor block_tables;
    infinicore::Tensor seq_lens;
    infinilm::global_state::FlashMLASchedMeta scheduler_metadata;
};
```

前三个字段是 FlashMLA decode 的输入 metadata：

- `slot_mapping`：当前 token 写入 paged MLA cache 的 slot。
- `block_tables`：每个 request 的 paged cache block table。
- `seq_lens`：每个 request 的总 KV 长度，对应 Python 输入里的 `total_kv_lengths`。

`scheduler_metadata` 是 FlashMLA 内部 tile scheduler 的缓存，类型是 `FlashMLASchedMeta`，定义在 `csrc/global_state/flash_mla_sched_meta.hpp`：

```cpp
struct FlashMLASchedMeta {
    bool have_initialized{false};
    std::optional<Config> config;
    infinicore::Tensor tile_scheduler_metadata;
    infinicore::Tensor num_splits;
};
```

其中真正参与 FlashMLA dense decode 复用的是：

- `tile_scheduler_metadata`
- `num_splits`

`config` 用来校验同一个 sched meta 被复用时，batch、query length、head 数、page block size、causal 等配置保持一致。

## Python Processor 阶段

`python/infinilm/processors/glm4_moe_lite_processor.py` 只生成基础 attention metadata，不生成 scheduler metadata。

生成的数据包括：

- `input_ids`
- `position_ids`
- `past_kv_lengths`
- `total_kv_lengths`
- `input_offsets`
- `cu_seqlens`
- `block_tables`
- `slot_mapping`
- sampling 参数

其中 FlashMLA 相关的是：

```python
"total_kv_lengths": ...
"block_tables": ...
"slot_mapping": ...
```

这里没有 `tile_scheduler_metadata` 和 `num_splits`，因为它们依赖 FlashMLA backend 根据实际 decode shape 计算。

## Input 转 ForwardContext 阶段

`csrc/engine/infer_engine.cpp` 的 `InferEngine::Input::to_model_input()` 会把 Python 传入的基础 metadata 绑定到 thread-local `ForwardContext`：

```cpp
forward_context.flashmla_attn_metadata = FlashMLAMetadata(
    input.slot_mapping.value_or(infinicore::Tensor()),
    input.block_tables.value_or(infinicore::Tensor()),
    input.total_sequence_lengths.value_or(infinicore::Tensor()));
```

注意：这里每次模型 forward 前都会重新构造一个新的 `FlashMLAMetadata` 对象，`scheduler_metadata` 默认是空的。

这意味着：

- eager 路径下，不会跨推理请求复用 scheduler metadata。
- graph replay 路径下，需要由 graph compiler 显式重新绑定已缓存的 scheduler metadata。

## GLM Attention 中的 prefill / decode 分流

`csrc/models/glm4_moe_lite/glm4_moe_lite_attention.cpp` 中，GLM 根据 token 数和 request 数判断 prefill 还是 decode：

```cpp
const auto &flashmla_attn_metadata = forward_context.flashmla_attn_metadata;
const size_t num_requests = flashmla_attn_metadata.seq_lens->numel();
const bool is_prefill = tokens != num_requests;
```

### Prefill

prefill 走 `forward_mha()`：

```cpp
auto output = forward_mha(q, kv_c, k_pe, tokens);
```

prefill 不调用 FlashMLA dense decode，因此不使用 `scheduler_metadata`。

但 prefill 仍然会调用：

```cpp
mla_attn_->do_kv_cache_update(kv_c, k_pe);
```

这一步只负责把当前 prefill token 的 MLA latent KV 写入 paged MLA cache，供后续 decode 使用。

### Decode

decode 走 FlashMLA MQA 路径：

```cpp
auto [attn_latent_4d, lse] = mla_attn_->forward_mqa(q_flash, kv_c, k_pe);
```

`MLAAttentionLayer::forward_mqa()` 会从 `ForwardContext` 中取当前 layer 的 KV cache 和 `flashmla_attn_metadata`，然后转发给 `FlashMLAImpl::forward_mqa()`。

## FlashMLAImpl::forward_mqa 中的使用

`FlashMLAImpl::forward_mqa()` 的流程是：

1. 检查 `query`、`kv_c`、`k_pe`、`kv_cache`。
2. 检查 `block_tables`、`seq_lens`、`slot_mapping` 是否存在。
3. 调用 `do_kv_cache_update()` 更新当前 layer 的 MLA KV cache。
4. 将 `[num_blocks, block_size, latent_dim]` 的 cache view 成 FlashMLA 需要的 4D cache。
5. 取出 `attn_metadata.scheduler_metadata`，传给 `flash_mla_with_kvcache()`。

关键代码形态：

```cpp
auto &scheduler_metadata = attn_metadata.scheduler_metadata;

return flash_mla_with_kvcache(
    query,
    kv_cache_4d,
    attn_metadata.block_tables,
    attn_metadata.seq_lens,
    head_dim_v_,
    scheduler_metadata,
    ...);
```

这里传入的是引用，因此 `flash_mla_with_kvcache()` 内部对 `scheduler_metadata` 的写入会直接保存在当前 `ForwardContext::flashmla_attn_metadata` 中。

## flash_mla_with_kvcache 中的 dense scheduler 生成和复用

GLM4 MoE Lite 当前使用 dense FlashMLA decode，不传 sparse indices，因此进入 dense 分支。

dense 分支先判断是否已有 scheduler tensor：

```cpp
const bool has_schedule =
    sched_meta.tile_scheduler_metadata && sched_meta.num_splits;
```

### 第一次调用

第一次进入某个推理的 decode forward 时，`has_schedule == false`，于是传给 InfiniCore 的是 `std::nullopt`：

```cpp
decode_tile_scheduler_metadata = std::nullopt;
decode_num_splits = std::nullopt;
```

随后调用：

```cpp
auto [out, lse, new_tile_scheduler_metadata, new_num_splits] =
    infinicore::op::flash_mla::dense_decode_fwd(...);
```

InfiniCore / FlashMLA vendor 会根据当前 batch、q shape、block table、seq lens 等信息生成 scheduler metadata，并返回两个 `infinicore::Tensor`：

- `new_tile_scheduler_metadata`
- `new_num_splits`

InfiniLM 直接保存这两个 Tensor：

```cpp
sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata;
sched_meta.num_splits = new_num_splits;
sched_meta.have_initialized = true;
```

这里不需要额外 D2D copy，因为 `dense_decode_fwd()` 返回的已经是 `infinicore::Tensor` 对象。

### 后续 decoder layer

同一次 decode forward 中，后续 decoder layer 共享同一个 `ForwardContext::flashmla_attn_metadata.scheduler_metadata`。

因此第二层及之后：

```cpp
has_schedule == true
```

FlashMLA dense decode 会收到已有的：

```cpp
sched_meta.tile_scheduler_metadata
sched_meta.num_splits
```

这样 47 个 decoder layer 只需要第一层生成一次 scheduler metadata，后续层复用。

这个复用是合理的，因为 scheduler metadata 只依赖 decode batch 形态、cache page layout 和 seq lens，不依赖 layer 权重。

## Eager 路径生命周期

普通 eager 推理中，生命周期如下：

1. Python processor 生成 `slot_mapping`、`block_tables`、`total_kv_lengths` 等基础 metadata。
2. `InferEngine::Input::to_model_input()` 每次 forward 创建新的 `FlashMLAMetadata`，其中 scheduler metadata 为空。
3. GLM decode 第一层调用 FlashMLA dense decode，生成 scheduler metadata。
4. 当前 decode forward 的后续 decoder layer 复用这份 scheduler metadata。
5. 本次 forward 结束后，这份 metadata 不跨下一次推理自动复用。

也就是说，eager 下的复用范围是：

```text
一次模型 forward 内的多个 decoder layer
```

不是跨请求、跨 decode step 的全局复用。

## Graph compile 路径

graph compile 需要提前准备 scheduler metadata，因为 graph capture/replay 不能依赖运行时再分配 scheduler tensor。

`csrc/engine/compiler/paged_compiler.cpp` 中，每个 decode batch size 的 compile 流程是：

1. `make_decode_input(b)` 构造 decode graph input，并设置：

```cpp
forward_context.flashmla_attn_metadata = FlashMLAMetadata(
    input.slot_mapping.value(),
    input.block_tables.value(),
    input.total_sequence_lengths.value());
```

2. graph capture 前先跑一次 eager forward：

```cpp
(void)model_->forward(input);
infinicore::context::syncStream();
```

这一次 eager forward 会在 GLM 第一层 FlashMLA dense decode 时生成 scheduler metadata。

3. 从 forward context 中收集生成后的 scheduler metadata：

```cpp
auto flashmla_sched_meta_vec =
    collect_flashmla_sched_meta_vec_from_forward_context();
```

`collect_flashmla_sched_meta_vec_from_forward_context()` 只有在：

```cpp
flashmla_attn_metadata.has_sched_meta()
```

为 true 时才返回 metadata。

4. 将 scheduler metadata 重新绑定回 graph input 对应的 forward context：

```cpp
bind_flashmla_forward_context_from_input(input, flashmla_sched_meta_vec);
```

5. 开始 graph recording：

```cpp
infinicore::context::startGraphRecording();
auto output = model_->forward(input);
auto graph = infinicore::context::stopGraphRecording();
```

capture 时 FlashMLA dense decode 已经能看到已有的 `tile_scheduler_metadata` 和 `num_splits`，不会走运行时生成 metadata 的路径。

6. `CompiledResult` 保存：

```cpp
std::vector<FlashMLASchedMeta> flashmla_sched_meta_vec;
```

当前 GLM 只有一种 FlashMLA attention 类型，因此 vector size 为 1。

## Graph replay 路径

decode graph replay 时，`PagedCompiler::get_compiled()` 根据 runtime batch size 找到对应 `CompiledResult`。

在 replay 前，它会把 runtime 输入复制到 graph input，然后重新绑定 FlashMLA scheduler metadata：

```cpp
if (!result->second.flashmla_sched_meta_vec.empty()) {
    bind_flashmla_forward_context_from_input(
        graph_input,
        result->second.flashmla_sched_meta_vec);
}
```

绑定后：

- `slot_mapping`
- `block_tables`
- `seq_lens`
- `scheduler_metadata`

都会挂到当前 thread-local `ForwardContext::flashmla_attn_metadata` 上。

随后 `graph->run()` 执行 graph replay。FlashMLA dense decode 在 graph 中复用 capture 时绑定的 scheduler tensors。

## 为什么不能跨推理复用

`scheduler_metadata` 中的 `config` 会记录：

- batch size
- query length
- q heads
- page block size
- k heads
- causal
- sparse topk 相关配置

这些配置必须和当前 decode 调用一致。不同请求、不同 batch、不同 seq lens / block table 场景下，scheduler metadata 可能不兼容。

当前设计中：

- eager：每次 forward 重新构造 `FlashMLAMetadata`，避免跨推理误复用。
- graph：每个 compiled batch size 保存自己的 `FlashMLASchedMeta`，replay 时按 batch size 绑定。

因此 scheduler metadata 的安全复用边界是：

```text
eager: 同一次 forward 内的多个 decoder layer
graph: 同一个 compiled batch graph 的 capture/replay
```

## 小结

GLM4 MoE Lite 中 `FlashMLAMetadata::scheduler_metadata` 的核心作用是缓存 FlashMLA dense decode 的 tile scheduling 结果。

完整流向如下：

```text
Python processor
  只生成 slot_mapping / block_tables / total_kv_lengths

InferEngine::Input::to_model_input
  构造 FlashMLAMetadata，scheduler_metadata 初始为空

GLM decode first layer
  FlashMLA dense_decode_fwd 生成 tile_scheduler_metadata / num_splits

GLM decode later layers
  复用同一个 scheduler_metadata

PagedCompiler graph compile
  eager forward 预生成 scheduler_metadata
  capture 前绑定到 forward context
  CompiledResult 保存 flashmla_sched_meta_vec

PagedCompiler graph replay
  根据 batch size 找 compiled result
  replay 前重新绑定 scheduler_metadata
  graph 中直接复用
```

这个设计避免了每个 decoder layer 都重新构建 FlashMLA schedule，也避免 eager 请求之间错误复用 scheduler metadata。
