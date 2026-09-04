#pragma once

#include "flashmla.hpp"
#include "infinicore/tensor.hpp"

#include <cstdint>
#include <optional>
#include <utility>

namespace infinilm::layers::mla_attention {

class MLAAttentionLayer;

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
    std::optional<infinicore::Tensor> cp_tot_seqlen_k);

namespace backends {

class FlashMLAV2Impl {
public:
    FlashMLAV2Impl(size_t num_heads,
                   size_t head_size,
                   float scale,
                   size_t num_kv_heads,
                   size_t layer_idx,
                   size_t head_dim_v);

    std::pair<infinicore::Tensor, infinicore::Tensor> forward_mqa(
        const MLAAttentionLayer &layer,
        const infinicore::Tensor &query,
        const infinicore::Tensor &kv_c,
        const infinicore::Tensor &k_pe,
        infinicore::Tensor &kv_cache,
        FlashMLAMetadata &attn_metadata) const;

    void do_kv_cache_update(const infinicore::Tensor &kv_c,
                            const infinicore::Tensor &k_pe,
                            infinicore::Tensor &kv_cache,
                            const infinicore::Tensor &slot_mapping) const;

private:
    size_t num_heads_;
    size_t head_size_;
    float scale_;
    size_t num_kv_heads_;
    size_t layer_idx_;
    size_t head_dim_v_;
};

} // namespace backends
} // namespace infinilm::layers::mla_attention
