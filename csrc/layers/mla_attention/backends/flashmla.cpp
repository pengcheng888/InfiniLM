#include "flashmla.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/concat_and_cache_mla.hpp"
#include "infinicore/ops/flash_mla/dense_decode_fwd.hpp"
#include "infinicore/ops/flash_mla/sparse_decode_fwd.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <stdexcept>

namespace infinilm::layers::mla_attention {

#define FLASH_MLA_SCHED_META_HELPER_MSG                        \
    " Your input arguments are inconsistent with sched_meta. " \
    "Please make sure the input arguments are consistent across different invocations of flash_mla_with_kvcache on the same sched_meta."

std::pair<infinicore::Tensor, infinicore::Tensor> flash_mla_with_kvcache(
    const infinicore::Tensor &q,
    const infinicore::Tensor &k_cache,
    std::optional<infinicore::Tensor> block_table,
    std::optional<infinicore::Tensor> cache_seqlens,
    int64_t head_dim_v,
    infinilm::global_state::FlashMLASchedMeta &tile_scheduler_metadata,
    std::optional<infinicore::Tensor> num_splits,
    std::optional<double> softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    std::optional<infinicore::Tensor> indices,
    std::optional<infinicore::Tensor> attn_sink,
    std::optional<infinicore::Tensor> extra_k_cache,
    std::optional<infinicore::Tensor> extra_indices_in_kvcache,
    std::optional<infinicore::Tensor> topk_length,
    std::optional<infinicore::Tensor> extra_topk_length) {

    auto &sched_meta = tile_scheduler_metadata;
    auto indices_in_kvcache = indices;

    ASSERT(q);
    ASSERT(k_cache);
    ASSERT(q->ndim() == 4);
    ASSERT(k_cache->ndim() == 4);

    std::optional<size_t> topk;
    if (indices_in_kvcache.has_value() && indices_in_kvcache.value()) {
        ASSERT(indices_in_kvcache.value()->ndim() > 0);
        topk = indices_in_kvcache.value()->size(indices_in_kvcache.value()->ndim() - 1);
    }

    std::optional<size_t> extra_k_page_block_size;
    if (extra_k_cache.has_value() && extra_k_cache.value()) {
        ASSERT(extra_k_cache.value()->ndim() >= 2);
        extra_k_page_block_size = extra_k_cache.value()->size(1);
    }

    std::optional<size_t> extra_topk;
    if (extra_indices_in_kvcache.has_value() && extra_indices_in_kvcache.value()) {
        ASSERT(extra_indices_in_kvcache.value()->ndim() > 0);
        extra_topk = extra_indices_in_kvcache.value()->size(extra_indices_in_kvcache.value()->ndim() - 1);
    }

    const double scale = softmax_scale.has_value() ? softmax_scale.value() : 1.0 / std::sqrt(static_cast<double>(q->size(q->ndim() - 1)));

    const bool had_initialized = sched_meta.have_initialized;
    if (!had_initialized) {
        // Sanity check.We only perform sanity check during the first invocation to save CPU time.
        if (indices_in_kvcache.has_value()) {
            ASSERT(!causal && "causal must be False when indices_in_kvcache is not None (i.e. sparse attention is enabled)");
        }
        sched_meta.have_initialized = true;
        sched_meta.config = infinilm::global_state::FlashMLASchedMeta::Config{
            q->size(0),
            q->size(1),
            q->size(2),
            k_cache->size(1),
            k_cache->size(2),
            causal,
            is_fp8_kvcache,
            topk,
            extra_k_page_block_size,
            extra_topk,
        };
    } else {
        ASSERT(sched_meta.config.has_value());
        const auto &config = sched_meta.config.value();
        ASSERT(config.b == q->size(0)
               && "sched_meta.config.b must be equal to batch_size." FLASH_MLA_SCHED_META_HELPER_MSG);
        ASSERT(config.s_q == q->size(1)
               && "sched_meta.config.s_q must be equal to seq_len_q." FLASH_MLA_SCHED_META_HELPER_MSG);
        ASSERT(config.h_q == q->size(2)
               && "sched_meta.config.h_q must be equal to num_heads_q." FLASH_MLA_SCHED_META_HELPER_MSG);
        ASSERT(config.page_block_size == k_cache->size(1)
               && "sched_meta.config.page_block_size must be equal to page_block_size." FLASH_MLA_SCHED_META_HELPER_MSG);
        ASSERT(config.h_k == k_cache->size(2)
               && "sched_meta.config.h_k must be equal to num_heads_k." FLASH_MLA_SCHED_META_HELPER_MSG);
        ASSERT(config.causal == causal
               && "sched_meta.config.causal must be equal to causal." FLASH_MLA_SCHED_META_HELPER_MSG);
        ASSERT(config.is_fp8_kvcache == is_fp8_kvcache
               && "sched_meta.config.is_fp8_kvcache must be equal to is_fp8_kvcache." FLASH_MLA_SCHED_META_HELPER_MSG);
        ASSERT(config.topk == topk
               && "sched_meta.config.topk must be equal to the last dim of indices_in_kvcache." FLASH_MLA_SCHED_META_HELPER_MSG);
        ASSERT(config.extra_page_block_size == extra_k_page_block_size
               && "sched_meta.config.extra_page_block_size must be equal to the page_block_size of extra_k_cache." FLASH_MLA_SCHED_META_HELPER_MSG);
        ASSERT(config.extra_topk == extra_topk
               && "sched_meta.config.extra_topk must be equal to the last dim of extra_indices_in_kvcache." FLASH_MLA_SCHED_META_HELPER_MSG);
    }

    if (topk.has_value()) {
        // Sparse attention
        if (causal) {
            ASSERT(false && "causal must be False when sparse attention is enabled");
        }

        if (!is_fp8_kvcache) {
            ASSERT((k_cache->dtype() == infinicore::DataType::BF16) && "BF16 sparse attention requires k_cache dtype to be bfloat16 when is_fp8_kvcache is False");

            if (extra_k_cache.has_value() && extra_k_cache.value()) {
                ASSERT((extra_k_cache.value()->dtype() == infinicore::DataType::BF16) && "BF16 sparse attention requires extra_k_cache dtype to be bfloat16 when is_fp8_kvcache is False");
            }
        } else {
            ASSERT(!(num_splits.has_value()) && "num_splits override is only supported by BF16 sparse decode");
        }

        std::optional<infinicore::Tensor> decode_tile_scheduler_metadata = had_initialized ? std::optional<infinicore::Tensor>(sched_meta.tile_scheduler_metadata) : std::nullopt;
        std::optional<infinicore::Tensor> decode_num_splits = (num_splits.has_value() && num_splits.value())
                                                                ? num_splits
                                                                : (had_initialized ? std::optional<infinicore::Tensor>(sched_meta.num_splits) : std::nullopt);

        auto [out, lse, new_tile_scheduler_metadata, new_num_splits] = infinicore::op::flash_mla::sparse_decode_fwd(q,
                                                                                                                    k_cache,
                                                                                                                    indices_in_kvcache.value(),
                                                                                                                    topk_length,
                                                                                                                    attn_sink,
                                                                                                                    decode_tile_scheduler_metadata,
                                                                                                                    decode_num_splits,
                                                                                                                    extra_k_cache,
                                                                                                                    extra_indices_in_kvcache,
                                                                                                                    extra_topk_length,
                                                                                                                    head_dim_v,
                                                                                                                    scale);
        sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata;
        sched_meta.num_splits = new_num_splits;
        return {out, lse};
    } else {
        // Dense attention
        if (num_splits.has_value() && num_splits.value()) {
            ASSERT(false && "num_splits override is only supported by BF16 sparse decode");
        }
        if ((indices_in_kvcache.has_value() && indices_in_kvcache.value())
            || (attn_sink.has_value() && attn_sink.value())
            || (extra_k_cache.has_value() && extra_k_cache.value())
            || (extra_indices_in_kvcache.has_value() && extra_indices_in_kvcache.value())
            || (topk_length.has_value() && topk_length.value())
            || (extra_topk_length.has_value() && extra_topk_length.value())) {
            ASSERT(false && "indices_in_kvcache, attn_sink, extra_k_cache, extra_indices_in_kvcache, topk_length and extra_topk_length must be None when dense attention is used.");
        }

        ASSERT((block_table.has_value() && block_table.value() && cache_seqlens.has_value() && cache_seqlens.value()) && "block_table and cache_seqlens must be provided when dense attention is used.");
        ASSERT(k_cache->size(1) == 64 && "flash_mla_with_kvcache dense attention requires page_block_size == 64");

        const bool has_schedule = sched_meta.tile_scheduler_metadata && sched_meta.num_splits;
        std::optional<infinicore::Tensor> decode_tile_scheduler_metadata = has_schedule ? std::optional<infinicore::Tensor>(sched_meta.tile_scheduler_metadata) : std::nullopt;
        std::optional<infinicore::Tensor> decode_num_splits = has_schedule ? std::optional<infinicore::Tensor>(sched_meta.num_splits) : std::nullopt;
        bool reuse_sched_metadata = {false};
        if (!reuse_sched_metadata) {
            decode_tile_scheduler_metadata = std::nullopt;
            decode_num_splits = std::nullopt;
        }
        auto [out, lse, new_tile_scheduler_metadata, new_num_splits] = infinicore::op::flash_mla::dense_decode_fwd(q,
                                                                                                                   k_cache,
                                                                                                                   head_dim_v,
                                                                                                                   cache_seqlens.value(),
                                                                                                                   block_table.value(),
                                                                                                                   scale,
                                                                                                                   causal,
                                                                                                                   decode_tile_scheduler_metadata,
                                                                                                                   decode_num_splits);
        if (!has_schedule) {
            if (!new_tile_scheduler_metadata || !new_num_splits) {
                throw std::runtime_error("flash_mla_with_kvcache: empty FlashMLA dense schedule metadata");
            }
            if (new_tile_scheduler_metadata->dtype() != infinicore::DataType::I32
                || new_num_splits->dtype() != infinicore::DataType::I32) {
                throw std::runtime_error("flash_mla_with_kvcache: FlashMLA dense schedule metadata must be int32");
            }
            if (!new_tile_scheduler_metadata->is_contiguous()
                || !new_num_splits->is_contiguous()) {
                throw std::runtime_error("flash_mla_with_kvcache: FlashMLA dense schedule metadata must be contiguous");
            }
            sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata;
            sched_meta.num_splits = new_num_splits;
            sched_meta.have_initialized = true;
        }
        return {out, lse};
    }
}

#undef FLASH_MLA_SCHED_META_HELPER_MSG

namespace backends {

FlashMLAImpl::FlashMLAImpl(size_t num_heads,
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

std::pair<infinicore::Tensor, infinicore::Tensor> FlashMLAImpl::forward_mqa(
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
    ASSERT(kv_cache->size(1) == 64 && "FlashMLAImpl requires block_size == 64");
    ASSERT(num_kv_heads_ == 1 && "FlashMLAImpl currently supports MQA KV cache only");

    if (!attn_metadata.block_tables || !attn_metadata.seq_lens || !attn_metadata.slot_mapping) {
        throw std::runtime_error("FlashMLAImpl::forward_mqa requires FlashMLA attention metadata");
    }

    do_kv_cache_update(kv_c, k_pe, kv_cache, attn_metadata.slot_mapping);
    auto kv_cache_4d = kv_cache->view({kv_cache->size(0), kv_cache->size(1), 1, head_size_});

    auto &scheduler_metadata = attn_metadata.scheduler_metadata;

    const size_t query_tokens = query->size(0) * query->size(1);
    const bool causal = query_tokens > attn_metadata.seq_lens->numel();
    return flash_mla_with_kvcache(query,
                                  kv_cache_4d,
                                  attn_metadata.block_tables,
                                  attn_metadata.seq_lens,
                                  static_cast<int64_t>(head_dim_v_),
                                  scheduler_metadata,
                                  std::nullopt,
                                  static_cast<double>(scale_),
                                  causal,
                                  false,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt);
}

void FlashMLAImpl::do_kv_cache_update(const infinicore::Tensor &kv_c,
                                      const infinicore::Tensor &k_pe,
                                      infinicore::Tensor &kv_cache,
                                      const infinicore::Tensor &slot_mapping) const {
    ASSERT(kv_c);
    ASSERT(k_pe);
    ASSERT(kv_cache);
    ASSERT(slot_mapping);
    ASSERT(kv_cache->ndim() == 3);
    ASSERT(kv_cache->size(1) == 64 && "FlashMLAImpl requires block_size == 64");
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
