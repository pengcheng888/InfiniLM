#include "flashmla.hpp"

#include "../../../global_state/global_state.hpp"

#include "infinicore/ops/concat_and_cache_mla.hpp"
#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"

#include <optional>
#include <stdexcept>

namespace infinilm::layers::mla_attention::backends {

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
    const infinicore::Tensor &query,
    const infinicore::Tensor &kv_c,
    const infinicore::Tensor &k_pe) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.attn_metadata;
    if (forward_context.kv_cache_vec.size() <= layer_idx_
        || !forward_context.kv_cache_vec[layer_idx_]) {
        throw std::runtime_error("FlashMLAImpl::forward_mqa requires MLA KV cache allocation");
    }
    if (!attn_metadata.total_sequence_lengths || !attn_metadata.block_tables
        || !attn_metadata.slot_mapping) {
        throw std::runtime_error("FlashMLAImpl::forward_mqa requires paged attention metadata");
    }
    if (!query || query->ndim() != 4 || query->size(1) != 1
        || query->size(2) != num_heads_ || query->size(3) != head_size_) {
        throw std::runtime_error("FlashMLAImpl::forward_mqa expects decode query [batch, 1, heads, head_size]");
    }
    if (num_kv_heads_ != 1) {
        throw std::runtime_error("FlashMLAImpl::forward_mqa currently supports one MLA KV head");
    }

    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
    do_kv_cache_update(kv_c, k_pe, kv_cache, attn_metadata.slot_mapping.value());
    if (kv_cache->ndim() != 3 || kv_cache->size(1) != 64
        || kv_cache->size(2) != head_size_) {
        throw std::runtime_error("FlashMLAImpl::forward_mqa expects KV cache [blocks, 64, head_size]");
    }
    auto kv_cache_4d = kv_cache->view({kv_cache->size(0), kv_cache->size(1), 1, head_size_});
    return infinicore::op::flash_mla::flash_mla_with_kvcache(
        query,
        kv_cache_4d,
        attn_metadata.block_tables.value(),
        attn_metadata.total_sequence_lengths.value(),
        static_cast<int64_t>(head_dim_v_),
        infinicore::op::flash_mla::FlashMLASchedMeta(),
        std::nullopt,
        static_cast<double>(scale_),
        false,
        false);
}

void FlashMLAImpl::do_kv_cache_update(const infinicore::Tensor &kv_c,
                                       const infinicore::Tensor &k_pe) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    if (forward_context.kv_cache_vec.size() <= layer_idx_
        || !forward_context.kv_cache_vec[layer_idx_]
        || !forward_context.attn_metadata.slot_mapping) {
        throw std::runtime_error("FlashMLAImpl::do_kv_cache_update requires cache and slot mapping");
    }
    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
    do_kv_cache_update(kv_c,
                       k_pe,
                       kv_cache,
                       forward_context.attn_metadata.slot_mapping.value());
}

void FlashMLAImpl::do_kv_cache_update(const infinicore::Tensor &kv_c,
                                       const infinicore::Tensor &k_pe,
                                       infinicore::Tensor &kv_cache,
                                       const infinicore::Tensor &slot_mapping) const {
    auto cache_scale = infinicore::Tensor::ones(
        {1}, infinicore::DataType::F32, kv_cache->device());
    infinicore::op::concat_and_cache_mla_(
        kv_c, k_pe, kv_cache, slot_mapping, "auto", cache_scale);
}

} // namespace infinilm::layers::mla_attention::backends
