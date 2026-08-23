# DeepSeek V4 FlashMLA Schedule 生命周期说明

## 背景

当前 InfiniLM 仓库中，`DeepseekV4Attention::compute_sparse_attention` 会从 `ForwardContext` 获取 FlashMLA schedule cache：

```cpp
forward_context.deepseek_v4_flashmla_schedule_cache
```

该 cache 的结构定义在 `csrc/global_state/forward_context.hpp`：

```cpp
struct DeepSeekV4FlashMLAScheduleCache {
    infinicore::Tensor swa_tile_scheduler_metadata;
    infinicore::Tensor swa_num_splits;
    infinicore::Tensor c4_tile_scheduler_metadata;
    infinicore::Tensor c4_num_splits;
    infinicore::Tensor c128_tile_scheduler_metadata;
    infinicore::Tensor c128_num_splits;
};
```

其中每类 attention 都有一组 FlashMLA schedule tensor：

- `tile_scheduler_metadata`
- `num_splits`

`compress_ratio_ == 0` 使用 SWA schedule；`compress_ratio_ == 4` 使用 C4 schedule；`compress_ratio_ == 128` 使用 C128 schedule。

## 当前流程

每次 forward/decode 开始时，eager 路径会在 `InferEngine::Input::to_model_input` 中将 `deepseek_v4_flashmla_schedule_cache` 重置为空。

同一次 forward 内：

1. 第一个同类 attention layer 发现 schedule cache 为空。
2. 调用 `infinicore::op::deepseek_v4_flashmla_sparse_attention_with_metadata_(...)` 完成 attention 计算，并返回 `DeepseekV4FlashMLASparseAttentionSchedule`。
3. `DeepseekV4Attention::cache_flashmla_schedule_metadata` 将返回的 schedule 写入 `DeepSeekV4FlashMLAScheduleCache`。
4. 后续同类 attention layer 从 cache 中取出非空 schedule，优先调用 `deepseek_v4_flashmla_sparse_attention_out_workspace_(...)` 复用 schedule。

## 生命周期风险

FlashMLA 返回的 schedule tensor 来自 FlashMLA/ATen 路径。虽然 InfiniCore 的 `from_aten_tensor` 会通过 deleter 持有 ATen tensor owner，但 schedule 后续还会进入 InfiniCore graph 相关链路。

`GraphTensor` 会通过 `Tensor::to_blob_()` 记录 tensor 地址，blob 本身不拥有原始 storage。因此 graph capture/replay 阶段要求被记录的 schedule tensor 在 graph 生命周期内具有稳定 owner 和稳定地址。

如果直接把 FlashMLA/ATen 返回 tensor 句柄保存到 schedule cache：

```cpp
schedule_cache.swa_tile_scheduler_metadata = flashmla_schedule.tile_scheduler_metadata;
schedule_cache.swa_num_splits = flashmla_schedule.num_splits;
```

cache 会依赖 FlashMLA/ATen 返回对象的 owner 链路。为了让 graph 使用的 schedule 更可控，当前修复改为将 schedule 内容拷贝到由 InfiniCore allocator 分配、由 `DeepSeekV4FlashMLAScheduleCache` 持有的 tensor 中。

## 修复实现

在 `csrc/models/deepseek_v4/deepseek_v4_attention.cpp` 中新增 helper：

```cpp
void copy_flashmla_schedule_tensor(infinicore::Tensor &dst,
                                   const infinicore::Tensor &src);
```

该 helper 做以下工作：

1. 校验 `src` 必须存在。
2. 校验 `src` 的 dtype 必须是 `infinicore::DataType::I32`。
3. 校验 `src` 必须连续。
4. 如果 `dst` 为空，或 shape/dtype/device 与 `src` 不一致，则使用 `infinicore::Tensor::empty(...)` 重新分配 `dst`。
5. 使用 `infinicore::context::memcpyD2D(dst->data(), src->data(), bytes, false)` 同步拷贝 schedule 内容。

`cache_flashmla_schedule_metadata` 中的回填逻辑改为：

```cpp
if (!schedule_cache.swa_tile_scheduler_metadata && flashmla_schedule.tile_scheduler_metadata) {
    copy_flashmla_schedule_tensor(schedule_cache.swa_tile_scheduler_metadata,
                                  flashmla_schedule.tile_scheduler_metadata);
}
if (!schedule_cache.swa_num_splits && flashmla_schedule.num_splits) {
    copy_flashmla_schedule_tensor(schedule_cache.swa_num_splits,
                                  flashmla_schedule.num_splits);
}
```

C4 和 C128 的回填逻辑同理。

该拷贝只发生在同一次 forward 中某类 attention 首次生成 schedule 时，不在后续同类 layer 的 `out_workspace_` 热路径中。

## Graph 路径

`PagedCompiler` 的 graph capture 流程会先运行一次 eager forward：

```cpp
(void)model_->forward(input);
```

这一步会生成 `DeepSeekV4FlashMLAScheduleCache`。随后 compiler 从 `ForwardContext` 中取出该 cache，并通过 `bind_graph_forward_context_from_input(input, deepseek_v4_flashmla_schedule_cache)` 绑定到 graph recording 使用的 forward context。

graph recording 开始后，`PagedCompiler::refresh_deepseek_v4_flashmla_schedules` 会调用：

```cpp
infinicore::op::deepseek_v4_flashmla_sparse_attention_metadata_(...)
```

它会在已经存在的 schedule tensor storage 上刷新：

- SWA schedule
- C4 schedule
- C128 schedule

因此 graph 路径依赖一个前提：eager forward 后 schedule cache 中的 tensor 必须已经存在，且 storage 必须稳定。当前修复正是为了满足这个前提。

## 注意事项

`compute_sparse_attention` 中不能把 `flashmla_tile_scheduler_metadata_opt` 和 `flashmla_num_splits_opt` 强制置为 `std::nullopt`。

graph recording 阶段需要看到非空 schedule，才能让 `deepseek_v4_flashmla_sparse_attention_with_metadata_(...)` 进入 graph op `execute` 路径。如果 recording 阶段传入空 schedule，FlashMLA attention 可能走 eager impl，attention 节点不会被正确记录到 InfiniCore graph 中。

## 修改结果

当前实现效果：

- FlashMLA schedule cache 存在于 `DeepSeekV4FlashMLAScheduleCache`，不是 `DeepSeekV4AttentionMetadata`。
- 首次生成 schedule 后，`DeepseekV4Attention` 将返回 schedule 拷贝到 cache-owned InfiniCore tensor。
- graph capture 使用 cache-owned schedule tensor，并在 recording 开头用 `deepseek_v4_flashmla_sparse_attention_metadata_` 刷新内容。
- 后续同一 forward 内的同类 attention layer 继续复用 schedule，并优先走 `deepseek_v4_flashmla_sparse_attention_out_workspace_(...)`。
- 本修改不改变正常推理数值，只改变 schedule cache 的 owner/lifetime。
