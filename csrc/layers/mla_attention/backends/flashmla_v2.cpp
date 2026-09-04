#include "flashmla_v2.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/concat_and_cache_mla.hpp"
#include "infinicore/ops/flash_mla/fwd_kvcache_mla.hpp"
#include "infinicore/ops/flash_mla/get_mla_decoding_metadata.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <stdexcept>

namespace infinilm::layers::mla_attention {

#define FLASH_MLA_V2_SCHED_META_HELPER_MSG                     \
    " Your input arguments are inconsistent with sched_meta. " \
    "Please make sure the input arguments are consistent across different invocations of flash_mla_v2_with_kvcache on the same sched_meta."

std::pair<infinicore::Tensor, infinicore::Tensor> flash_mla_v2_with_kvcache(
    const infinicore::Tensor &q,
    const infinicore::Tensor &k_cache,
    const infinicore::Tensor &block_table,
    const infinicore::Tensor &cache_seqlens,
    int64_t head_dim_v,
    const infinicore::Tensor &tile_scheduler_metadata,
    const infinicore::Tensor &num_splits,
    std::optional<double> softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    std::optional<infinicore::Tensor> indices,
    std::optional<infinicore::Tensor> indices_all_valid_per_q,
    std::optional<infinicore::Tensor> descale_q,
    std::optional<infinicore::Tensor> descale_k,
    int64_t cp_world_size,
    int64_t cp_rank,
    std::optional<infinicore::Tensor> cp_tot_seqlen_k) {

    ASSERT(q);
    ASSERT(k_cache);
    ASSERT(block_table);
    ASSERT(cache_seqlens);
    ASSERT(tile_scheduler_metadata);
    ASSERT(num_splits);
    ASSERT(q->ndim() == 4);
    ASSERT(k_cache->ndim() == 4);

    std::optional<size_t> topk;
    if (indices.has_value() && indices.value()) {
        ASSERT(indices.value()->ndim() > 0);
        topk = indices.value()->size(indices.value()->ndim() - 1);
    }

    const double scale = softmax_scale.has_value()
                           ? softmax_scale.value()
                           : 1.0 / std::sqrt(static_cast<double>(q->size(q->ndim() - 1)));
    ASSERT((descale_q.has_value() && descale_q.value()) == (descale_k.has_value() && descale_k.value())
           && "descale_q and descale_k should be both None or both not None");

    if (topk.has_value()) {
        throw std::runtime_error("flash_mla_v2_with_kvcache does not support sparse decode yet.");
    }
    if ((indices.has_value() && indices.value())
        || (indices_all_valid_per_q.has_value() && indices_all_valid_per_q.value())) {
        throw std::runtime_error("flash_mla_v2_with_kvcache dense path requires sparse arguments to be None.");
    }

    ASSERT(k_cache->size(1) == 64 && "flash_mla_v2_with_kvcache dense attention requires page_block_size == 64");

    infinicore::Tensor out = infinicore::Tensor::empty({q->size(0), q->size(1), q->size(2), static_cast<size_t>(head_dim_v)},
                                                       q->dtype(),
                                                       q->device());
    infinicore::Tensor lse = infinicore::Tensor::empty({q->size(0), q->size(2), q->size(1)},
                                                       infinicore::DataType::F32,
                                                       q->device());

    infinicore::op::flash_mla::fwd_kvcache_mla_(out,
                                                lse,
                                                q,
                                                k_cache,
                                                std::nullopt,
                                                head_dim_v,
                                                cache_seqlens,
                                                block_table,
                                                scale,
                                                causal,
                                                tile_scheduler_metadata,
                                                num_splits,
                                                is_fp8_kvcache,
                                                indices,
                                                indices_all_valid_per_q,
                                                cp_world_size,
                                                cp_rank,
                                                cp_tot_seqlen_k);

    return {out, lse};
}

namespace backends {

FlashMLAV2Impl::FlashMLAV2Impl(size_t num_heads,
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

std::pair<infinicore::Tensor, infinicore::Tensor> FlashMLAV2Impl::forward_mqa(
    const MLAAttentionLayer &layer,
    const infinicore::Tensor &query,
    const infinicore::Tensor &kv_c,
    const infinicore::Tensor &k_pe,
    infinicore::Tensor &kv_cache,
    FlashMLAMetadata &attn_metadata) const {
    (void)layer;
    ASSERT(query);
    ASSERT(kv_c);
    ASSERT(k_pe);
    ASSERT(kv_cache);
    ASSERT(query->ndim() == 4);
    ASSERT(kv_cache->ndim() == 3);
    ASSERT(kv_cache->size(1) == 64 && "FlashMLAV2Impl requires block_size == 64");
    ASSERT(num_kv_heads_ == 1 && "FlashMLAV2Impl currently supports MQA KV cache only");

    if (!attn_metadata.block_tables || !attn_metadata.seq_lens || !attn_metadata.slot_mapping) {
        throw std::runtime_error("FlashMLAV2Impl::forward_mqa requires FlashMLA attention metadata");
    }

    do_kv_cache_update(kv_c, k_pe, kv_cache, attn_metadata.slot_mapping);
    auto kv_cache_4d = kv_cache->view({kv_cache->size(0), kv_cache->size(1), 1, head_size_});

    const size_t query_tokens = query->size(0) * query->size(1);
    const bool causal = query_tokens > attn_metadata.seq_lens->numel();

    auto &sched_meta = attn_metadata.scheduler_metadata;
    const std::optional<size_t> topk = std::nullopt;
    const bool is_fp8_kvcache = false;
    if (!sched_meta.have_initialized) {
        sched_meta.have_initialized = true;
        sched_meta.config = infinilm::global_state::FlashMLASchedMeta::Config{
            query->size(0),
            query->size(1),
            query->size(2),
            kv_cache_4d->size(1),
            kv_cache_4d->size(2),
            causal,
            is_fp8_kvcache,
            topk,
            std::nullopt,
            std::nullopt,
        };
    } else {
        ASSERT(sched_meta.config.has_value());
        const auto &config = sched_meta.config.value();
        ASSERT(config.b == query->size(0)
               && "sched_meta.config.b must be equal to batch_size." FLASH_MLA_V2_SCHED_META_HELPER_MSG);
        ASSERT(config.s_q == query->size(1)
               && "sched_meta.config.s_q must be equal to seq_len_q." FLASH_MLA_V2_SCHED_META_HELPER_MSG);
        ASSERT(config.h_q == query->size(2)
               && "sched_meta.config.h_q must be equal to num_heads_q." FLASH_MLA_V2_SCHED_META_HELPER_MSG);
        ASSERT(config.page_block_size == kv_cache_4d->size(1)
               && "sched_meta.config.page_block_size must be equal to page_block_size." FLASH_MLA_V2_SCHED_META_HELPER_MSG);
        ASSERT(config.h_k == kv_cache_4d->size(2)
               && "sched_meta.config.h_k must be equal to num_heads_k." FLASH_MLA_V2_SCHED_META_HELPER_MSG);
        ASSERT(config.causal == causal
               && "sched_meta.config.causal must be equal to causal." FLASH_MLA_V2_SCHED_META_HELPER_MSG);
        ASSERT(config.is_fp8_kvcache == is_fp8_kvcache
               && "sched_meta.config.is_fp8_kvcache must be equal to is_fp8_kvcache." FLASH_MLA_V2_SCHED_META_HELPER_MSG);
        ASSERT(config.topk == topk
               && "sched_meta.config.topk must be equal to the last dim of indices." FLASH_MLA_V2_SCHED_META_HELPER_MSG);
        ASSERT(!config.extra_page_block_size.has_value()
               && "sched_meta.config.extra_page_block_size must be empty for flash_mla_v2_with_kvcache." FLASH_MLA_V2_SCHED_META_HELPER_MSG);
        ASSERT(!config.extra_topk.has_value()
               && "sched_meta.config.extra_topk must be empty for flash_mla_v2_with_kvcache." FLASH_MLA_V2_SCHED_META_HELPER_MSG);
    }

    const bool is_graph_recording = infinicore::context::isGraphRecording();
    if (is_graph_recording) {
        ASSERT(sched_meta.has_sched_buffer() && "FlashMLA v2 graph mode requires preallocated scheduler metadata buffers.");
    }

    const int64_t kv_heads = static_cast<int64_t>(kv_cache_4d->size(2));
    ASSERT(kv_heads > 0 && "FlashMLAV2Impl requires positive kv heads.");
    const int64_t q_tokens_times_heads = static_cast<int64_t>(query->size(1) * query->size(2));
    ASSERT(q_tokens_times_heads % kv_heads == 0 && "FlashMLAV2Impl requires q tokens * q heads to be divisible by kv heads.");
    const int64_t num_q_tokens_per_head_k = q_tokens_times_heads / kv_heads;

    if (!sched_meta.have_refreshed) {
        auto [new_tile_scheduler_metadata, new_num_splits] = infinicore::op::flash_mla::get_mla_decoding_metadata(sched_meta.tile_scheduler_metadata,
                                                                                                                  sched_meta.num_splits,
                                                                                                                  attn_metadata.seq_lens,
                                                                                                                  num_q_tokens_per_head_k,
                                                                                                                  kv_heads,
                                                                                                                  std::nullopt,
                                                                                                                  is_fp8_kvcache,
                                                                                                                  std::nullopt);
        sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata;
        sched_meta.num_splits = new_num_splits;
        sched_meta.have_refreshed = true;
    }

    if (!sched_meta.tile_scheduler_metadata || !sched_meta.num_splits) {
        throw std::runtime_error("FlashMLAV2Impl::forward_mqa failed to build FlashMLA scheduler metadata.");
    }
    return flash_mla_v2_with_kvcache(query,
                                     kv_cache_4d,
                                     attn_metadata.block_tables,
                                     attn_metadata.seq_lens,
                                     static_cast<int64_t>(head_dim_v_),
                                     sched_meta.tile_scheduler_metadata,
                                     sched_meta.num_splits,
                                     static_cast<double>(scale_),
                                     causal,
                                     is_fp8_kvcache,
                                     std::nullopt,
                                     std::nullopt,
                                     std::nullopt,
                                     std::nullopt,
                                     1,
                                     0,
                                     std::nullopt);
}

#undef FLASH_MLA_V2_SCHED_META_HELPER_MSG

void FlashMLAV2Impl::do_kv_cache_update(const infinicore::Tensor &kv_c,
                                        const infinicore::Tensor &k_pe,
                                        infinicore::Tensor &kv_cache,
                                        const infinicore::Tensor &slot_mapping) const {
    ASSERT(kv_c);
    ASSERT(k_pe);
    ASSERT(kv_cache);
    ASSERT(slot_mapping);
    ASSERT(kv_cache->ndim() == 3);
    ASSERT(kv_cache->size(1) == 64 && "FlashMLAV2Impl requires block_size == 64");
    auto cache_scale = infinicore::Tensor::ones({1}, infinicore::DataType::F32, kv_cache->device());
    infinicore::op::concat_and_cache_mla_(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        "auto",
        cache_scale);
}

} // namespace backends
} // namespace infinilm::layers::mla_attention
