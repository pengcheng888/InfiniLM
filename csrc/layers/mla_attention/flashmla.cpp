#include "flashmla.hpp"

#include "../../cache/kv_cache.hpp"
#include "../../global_state/global_state.hpp"

#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"
#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace infinilm::layers::mla_attention {

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

infinicore::Tensor DenseFlashMLAImpl::create_layer_kv_cache(
    const cache::PagedKVCacheConfig &cache_config,
    size_t head_size,
    infinicore::DataType dtype,
    const infinicore::Device &device) {
    return infinicore::Tensor::empty({cache_config.num_blocks(), cache_config.block_size(), head_size},
                                     dtype,
                                     device);
}

std::pair<infinicore::Tensor, infinicore::Tensor> DenseFlashMLAImpl::forward(
    const infinicore::Tensor &query,
    infinicore::op::flash_mla::FlashMLASchedMeta &sched_meta) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.attn_metadata;
    if (forward_context.kv_cache_vec.size() <= layer_idx_ || !forward_context.kv_cache_vec[layer_idx_]) {
        throw std::runtime_error("DenseFlashMLAImpl::forward requires MLA KV cache allocation");
    }
    if (!attn_metadata.total_sequence_lengths || !attn_metadata.block_tables) {
        throw std::runtime_error("DenseFlashMLAImpl::forward requires paged attention metadata");
    }
    if (!query || query->ndim() != 4 || query->size(1) != 1 || query->size(2) != num_heads_ || query->size(3) != head_size_) {
        throw std::runtime_error("DenseFlashMLAImpl::forward expects decode query [batch, 1, heads, head_size]");
    }
    if (num_kv_heads_ != 1) {
        throw std::runtime_error("DenseFlashMLAImpl::forward currently supports one MLA KV head");
    }

    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
    if (kv_cache->ndim() != 3 || kv_cache->size(1) != 64 || kv_cache->size(2) != head_size_) {
        throw std::runtime_error("DenseFlashMLAImpl::forward expects KV cache [blocks, 64, head_size]");
    }

    // 注意：reuse_sched_meta的数值，不要影响下面的逻辑。
    if (!sched_meta.has_valid_sched_meta()) {
        // 以下注释不要删除：
        // 调用 get_mla_metadata_ 函数。
        // 该函数直接更新 sched_meta。

        // 提示：不同平台的get_mla_metadata_函数的实现不同，可能保持sched_meta为空；
        // 也可能会调用kernel，原地更新为一个有效的sched_meta。

        const auto num_q_tokens_per_head_k = static_cast<int64_t>(query->size(1) * query->size(2) / num_kv_heads_);
        infinicore::op::flash_mla::get_mla_metadata_(
            sched_meta,
            attn_metadata.total_sequence_lengths.value(),
            num_q_tokens_per_head_k,
            static_cast<int64_t>(num_kv_heads_),
            std::nullopt,
            false,
            std::nullopt);
    }

    // 以下注释不要删除：
    // 即使reuse_sched_meta为false，也允许上面尝试调用get_mla_metadata_；
    // 当前接受这部分额外开销，但flash_mla_with_kvcache仍会传入空metadata，
    // 因此不会复用sched_meta。

    constexpr bool reuse_sched_meta = true;
    infinicore::op::flash_mla::FlashMLASchedMeta empty_sched_meta;
    auto &current_sched_meta = [&]() -> infinicore::op::flash_mla::FlashMLASchedMeta & {
        if constexpr (reuse_sched_meta) {
            return sched_meta;
        }
        return empty_sched_meta;
    }();

    auto kv_cache_4d = kv_cache->view({kv_cache->size(0), kv_cache->size(1), 1, head_size_});
    return infinicore::op::flash_mla::flash_mla_with_kvcache(
        query,
        kv_cache_4d,
        attn_metadata.block_tables.value(),
        attn_metadata.total_sequence_lengths.value(),
        static_cast<int64_t>(head_dim_v_),
        current_sched_meta,
        std::nullopt,
        static_cast<double>(scale_),
        false,
        false);
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

std::pair<infinicore::Tensor, infinicore::Tensor> SparseFlashMLAImpl::forward(
    const infinicore::Tensor &query,
    const infinicore::Tensor &kv_cache,
    const infinicore::Tensor &indices,
    const infinicore::Tensor &attn_sink,
    const infinicore::Tensor &topk_lengths,
    infinicore::op::flash_mla::FlashMLASchedMeta &sched_meta) const {
    if (!query || query->ndim() != 4 || query->size(1) != 1
        || query->size(2) != num_heads_ || query->size(3) != head_size_) {
        throw std::runtime_error("SparseFlashMLAImpl::forward expects query [tokens, 1, heads, head_size]");
    }
    if (num_kv_heads_ != 1) {
        throw std::runtime_error("SparseFlashMLAImpl::forward currently supports one MLA KV head");
    }
    if (!kv_cache || kv_cache->ndim() != 4 || kv_cache->size(2) != num_kv_heads_
        || kv_cache->dtype() != infinicore::DataType::F8) {
        throw std::runtime_error("SparseFlashMLAImpl::forward expects FP8 KV cache [blocks, page_size, 1, cache_dim]");
    }
    if (!indices || indices->ndim() != 3 || indices->dtype() != infinicore::DataType::I32
        || indices->size(0) != query->size(0) || indices->size(1) != query->size(1)) {
        throw std::runtime_error("SparseFlashMLAImpl::forward expects indices [tokens, 1, topk]");
    }
    if (!topk_lengths || topk_lengths->ndim() != 1
        || topk_lengths->dtype() != infinicore::DataType::I32
        || topk_lengths->size(0) != query->size(0) * query->size(1)) {
        throw std::runtime_error("SparseFlashMLAImpl::forward expects topk_lengths [tokens]");
    }
    if (!attn_sink || attn_sink->ndim() != 1
        || attn_sink->dtype() != infinicore::DataType::F32
        || attn_sink->size(0) != num_heads_) {
        throw std::runtime_error("SparseFlashMLAImpl::forward expects FP32 attn_sink [heads]");
    }

    const bool has_tile_scheduler_metadata = static_cast<bool>(sched_meta.tile_scheduler_metadata);
    const bool has_num_splits = static_cast<bool>(sched_meta.num_splits);
    const size_t expected_num_splits = query->size(0) * query->size(1) + 1;
    if (has_tile_scheduler_metadata != has_num_splits
        || (has_tile_scheduler_metadata
            && (sched_meta.tile_scheduler_metadata->dtype() != infinicore::DataType::I32
                || sched_meta.tile_scheduler_metadata->ndim() != 2
                || sched_meta.tile_scheduler_metadata->size(1) != 8
                || !sched_meta.tile_scheduler_metadata->is_contiguous()
                || sched_meta.tile_scheduler_metadata->device() != query->device()
                || sched_meta.num_splits->dtype() != infinicore::DataType::I32
                || sched_meta.num_splits->ndim() != 1
                || sched_meta.num_splits->size(0) != expected_num_splits
                || !sched_meta.num_splits->is_contiguous()
                || sched_meta.num_splits->device() != query->device()))) {
        sched_meta = infinicore::op::flash_mla::FlashMLASchedMeta();
    }

    if (!sched_meta.has_valid_sched_meta()) {
        const auto num_q_tokens_per_head_k = static_cast<int64_t>(query->size(1) * query->size(2) / num_kv_heads_);
        infinicore::op::flash_mla::get_mla_metadata_(
            sched_meta,
            topk_lengths,
            num_q_tokens_per_head_k,
            static_cast<int64_t>(num_kv_heads_),
            static_cast<int64_t>(num_heads_),
            true,
            static_cast<int64_t>(indices->size(2)));
    }

    constexpr bool reuse_sched_meta = true;
    infinicore::op::flash_mla::FlashMLASchedMeta empty_sched_meta;
    auto &current_sched_meta = [&]() -> infinicore::op::flash_mla::FlashMLASchedMeta & {
        if constexpr (reuse_sched_meta) {
            return sched_meta;
        }
        return empty_sched_meta;
    }();

    return infinicore::op::flash_mla::flash_mla_with_kvcache(
        query,
        kv_cache,
        std::nullopt,
        std::nullopt,
        static_cast<int64_t>(head_dim_v_),
        current_sched_meta,
        std::nullopt,
        static_cast<double>(scale_),
        false,
        true,
        indices,
        attn_sink,
        std::nullopt,
        std::nullopt,
        topk_lengths,
        std::nullopt);
}

} // namespace infinilm::layers::mla_attention
