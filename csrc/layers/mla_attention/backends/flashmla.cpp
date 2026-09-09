#include "flashmla.hpp"

#include "../../../global_state/global_state.hpp"
#include "../../../utils.hpp"
#include "../flashmla.hpp"

#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"

#include <optional>
#include <stdexcept>

namespace infinilm::layers::mla_attention::backends {

DenseFlashMLAImpl::DenseFlashMLAImpl(size_t num_heads,
                                     size_t head_size,
                                     float scale,
                                     size_t num_kv_heads,
                                     size_t layer_idx,
                                     size_t head_dim_v)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      layer_idx_(layer_idx),
      head_dim_v_(head_dim_v) {}

std::pair<infinicore::Tensor, infinicore::Tensor> DenseFlashMLAImpl::forward_mqa(
    const infinicore::Tensor &query) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.attn_metadata;
    if (forward_context.kv_cache_vec.size() <= layer_idx_ || !forward_context.kv_cache_vec[layer_idx_]) {
        throw std::runtime_error("DenseFlashMLAImpl::forward_mqa requires MLA KV cache allocation");
    }
    if (!attn_metadata.total_sequence_lengths || !attn_metadata.block_tables) {
        throw std::runtime_error("DenseFlashMLAImpl::forward_mqa requires paged attention metadata");
    }
    if (!query || query->ndim() != 4 || query->size(1) != 1 || query->size(2) != num_heads_ || query->size(3) != head_size_) {
        throw std::runtime_error("DenseFlashMLAImpl::forward_mqa expects decode query [batch, 1, heads, head_size]");
    }
    if (num_kv_heads_ != 1) {
        throw std::runtime_error("DenseFlashMLAImpl::forward_mqa currently supports one MLA KV head");
    }

    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
    if (kv_cache->ndim() != 3 || kv_cache->size(1) != 64 || kv_cache->size(2) != head_size_) {
        throw std::runtime_error("DenseFlashMLAImpl::forward_mqa expects KV cache [blocks, 64, head_size]");
    }

    // 注意：reuse_sched_meta的数值，不要影响下面的逻辑。
    ASSERT(forward_context.sched_meta.sched_meta_vec.size() > 0);
    auto &global_sched_meta = forward_context.sched_meta.sched_meta_vec[0];
    if (!global_sched_meta.has_valid_sched_meta()) {
        // 以下注释不要删除：
        // 调用 get_mla_metadata 函数。
        // 如果 返回的new_sched_meta时有效的。就赋值给global_sched_meta。

        // 提示：不同平台的get_mla_metadata函数的实现不同，可能返回空sched_meta；
        // 也可能会调用kernel，计算一个有效的sched_meta。

        auto new_sched_meta = infinicore::op::flash_mla::get_mla_metadata();
        if (new_sched_meta.has_valid_sched_meta()) {
            global_sched_meta = new_sched_meta;
        }
    }

    // 以下注释不要删除：
    // 即使reuse_sched_meta为false，也允许上面尝试调用get_mla_metadata；
    // 当前接受这部分额外开销，但flash_mla_with_kvcache仍会传入空metadata，
    // 因此不会复用global_sched_meta。

    constexpr bool reuse_sched_meta = true;
    infinicore::op::flash_mla::FlashMLASchedMeta empty_sched_meta;
    auto &current_sched_meta = [&]() -> infinicore::op::flash_mla::FlashMLASchedMeta & {
        if constexpr (reuse_sched_meta) {
            return global_sched_meta;
        }
        return empty_sched_meta;
    }();

    auto kv_cache_4d = kv_cache->view({kv_cache->size(0), kv_cache->size(1), 1, head_size_});
    return infinilm::layers::mla_attention::compute_dense_flash_mla(
        query,
        kv_cache_4d,
        attn_metadata.block_tables.value(),
        attn_metadata.total_sequence_lengths.value(),
        head_dim_v_,
        current_sched_meta,
        scale_);
}

SparseFlashMLAImpl::SparseFlashMLAImpl(size_t num_heads,
                                       size_t head_size,
                                       float scale,
                                       size_t num_kv_heads,
                                       size_t head_dim_v)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      head_dim_v_(head_dim_v) {}

std::pair<infinicore::Tensor, infinicore::Tensor> SparseFlashMLAImpl::forward_mqa(
    const infinicore::Tensor &query,
    const infinicore::Tensor &kv_cache,
    const infinicore::Tensor &indices,
    const infinicore::Tensor &attn_sink,
    const infinicore::Tensor &topk_lengths,
    infinicore::op::flash_mla::FlashMLASchedMeta &sched_meta) const {
    if (!query || query->ndim() != 4 || query->size(1) != 1
        || query->size(2) != num_heads_ || query->size(3) != head_size_) {
        throw std::runtime_error("SparseFlashMLAImpl::forward_mqa expects query [tokens, 1, heads, head_size]");
    }
    if (num_kv_heads_ != 1) {
        throw std::runtime_error("SparseFlashMLAImpl::forward_mqa currently supports one MLA KV head");
    }
    if (!kv_cache || kv_cache->ndim() != 4 || kv_cache->size(2) != num_kv_heads_
        || kv_cache->dtype() != infinicore::DataType::F8) {
        throw std::runtime_error("SparseFlashMLAImpl::forward_mqa expects FP8 KV cache [blocks, page_size, 1, cache_dim]");
    }
    if (!indices || indices->ndim() != 3 || indices->dtype() != infinicore::DataType::I32
        || indices->size(0) != query->size(0) || indices->size(1) != query->size(1)) {
        throw std::runtime_error("SparseFlashMLAImpl::forward_mqa expects indices [tokens, 1, topk]");
    }
    if (!topk_lengths || topk_lengths->ndim() != 1
        || topk_lengths->dtype() != infinicore::DataType::I32
        || topk_lengths->size(0) != query->size(0) * query->size(1)) {
        throw std::runtime_error("SparseFlashMLAImpl::forward_mqa expects topk_lengths [tokens]");
    }
    if (!attn_sink || attn_sink->ndim() != 1
        || attn_sink->dtype() != infinicore::DataType::F32
        || attn_sink->size(0) != num_heads_) {
        throw std::runtime_error("SparseFlashMLAImpl::forward_mqa expects FP32 attn_sink [heads]");
    }

    return infinilm::layers::mla_attention::compute_sparse_flash_mla(
        query,
        kv_cache,
        indices,
        attn_sink,
        topk_lengths,
        head_dim_v_,
        sched_meta,
        scale_);
}

} // namespace infinilm::layers::mla_attention::backends
