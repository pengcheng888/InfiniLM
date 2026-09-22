#include "flashmla.hpp"

#include "../../global_state/global_state.hpp"

#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"
#include "infinicore/ops/flash_mla/get_mla_metadata.hpp"

#include <optional>
#include <stdexcept>

namespace infinilm::layers::mla_attention {

DenseFlashMLAImpl::DenseFlashMLAImpl(size_t num_heads,
                                     size_t head_size,
                                     float scale,
                                     size_t num_kv_heads,
                                     size_t head_dim_v)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      head_dim_v_(head_dim_v) {}

std::pair<infinicore::Tensor, infinicore::Tensor> DenseFlashMLAImpl::forward(
    const infinicore::Tensor &query,
    const DenseFlashMLACache &cache,
    infinicore::op::flash_mla::FlashMLASchedMeta &sched_meta) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.attn_metadata;
    if (!attn_metadata.total_sequence_lengths || !attn_metadata.block_tables) {
        throw std::runtime_error("DenseFlashMLAImpl::forward requires paged attention metadata");
    }
    if (!query || query->ndim() != 4 || query->size(1) != 1 || query->size(2) != num_heads_ || query->size(3) != head_size_) {
        throw std::runtime_error("DenseFlashMLAImpl::forward expects decode query [batch, 1, heads, head_size]");
    }
    if (num_kv_heads_ != 1) {
        throw std::runtime_error("DenseFlashMLAImpl::forward currently supports one MLA KV head");
    }

    const auto &kv_cache = cache.flashmla_cache_view();
    if (kv_cache->ndim() != 4 || kv_cache->size(1) != 64
        || kv_cache->size(2) != num_kv_heads_ || kv_cache->size(3) != head_size_) {
        throw std::runtime_error("DenseFlashMLAImpl::forward expects KV cache [blocks, 64, 1, head_size]");
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

    return infinicore::op::flash_mla::flash_mla_with_kvcache(
        query,
        kv_cache,
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
    const SparseFlashMLACache &cache,
    const infinicore::Tensor &indices,
    const infinicore::Tensor &attn_sink,
    const infinicore::Tensor &topk_lengths,
    infinicore::op::flash_mla::FlashMLASchedMeta &sched_meta) const {
    const auto &kv_cache = cache.flashmla_cache_view();
    if (!query || query->ndim() != 4 || query->size(1) != 1
        || query->size(2) != num_heads_ || query->size(3) != head_size_) {
        throw std::runtime_error("SparseFlashMLAImpl::forward expects query [tokens, 1, heads, head_size]");
    }
    if (num_kv_heads_ != 1) {
        throw std::runtime_error("SparseFlashMLAImpl::forward currently supports one MLA KV head");
    }
    if (!kv_cache || kv_cache->ndim() != 4 || kv_cache->size(2) != num_kv_heads_
        || (kv_cache->dtype() != infinicore::DataType::F8
            && kv_cache->dtype() != infinicore::DataType::BF16)) {
        throw std::runtime_error("SparseFlashMLAImpl::forward expects FP8 or BF16 KV cache [blocks, page_size, 1, cache_dim]");
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

    const bool is_fp8_kvcache = (kv_cache->dtype() == infinicore::DataType::F8);
    auto flash_query = query;
    if (!is_fp8_kvcache && kv_cache->size(3) != query->size(3)) {
        // MetaX BF16 FlashMLA 的 head 布局需要额外一段 rope：
        // cache 为 [原 head, rope 副本]，query 为 [nope, 0, rope]。
        // 中间 0 段对点积没有贡献，因此扩展后仍保持原始 QK score。
        const size_t rope_dim = kv_cache->size(3) - query->size(3);
        if (rope_dim == 0 || rope_dim >= query->size(3)) {
            throw std::runtime_error("SparseFlashMLAImpl cannot adapt query to BF16 cache head size");
        }
        const size_t nope_dim = query->size(3) - rope_dim;
        auto query_nope = query->narrow({{3, 0, nope_dim}});
        auto query_rope = query->narrow({{3, nope_dim, rope_dim}});
        auto query_pad = infinicore::Tensor::zeros({query->size(0), query->size(1), query->size(2), rope_dim},
                                                   query->dtype(),
                                                   query->device());
        flash_query = infinicore::op::cat({query_nope, query_pad, query_rope}, 3);
    }

    if (!sched_meta.has_valid_sched_meta()) {
        const auto num_q_tokens_per_head_k = static_cast<int64_t>(query->size(1) * query->size(2) / num_kv_heads_);
        infinicore::op::flash_mla::get_mla_metadata_(
            sched_meta,
            topk_lengths,
            num_q_tokens_per_head_k,
            static_cast<int64_t>(num_kv_heads_),
            static_cast<int64_t>(num_heads_),
            is_fp8_kvcache,
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
        flash_query,
        kv_cache,
        std::nullopt,
        std::nullopt,
        static_cast<int64_t>(head_dim_v_),
        current_sched_meta,
        std::nullopt,
        static_cast<double>(scale_),
        false,
        is_fp8_kvcache,
        indices,
        attn_sink,
        std::nullopt,
        std::nullopt,
        topk_lengths,
        std::nullopt);
}

} // namespace infinilm::layers::mla_attention
