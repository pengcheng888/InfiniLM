#pragma once

#include "../../../global_state/flash_mla_sched_meta.hpp"
#include "infinicore/tensor.hpp"

#include <cstdint>
#include <optional>
#include <utility>

namespace infinilm::layers::mla_attention {

class MLAAttentionLayer;

struct FlashMLAMetadata {
    infinicore::Tensor slot_mapping;
    infinicore::Tensor block_tables;
    infinicore::Tensor seq_lens;
    infinicore::Tensor input_offsets;
    infinicore::Tensor query_start_loc; // query_start_loc is equivalent to cu_seqlens.
    size_t max_query_len{0};
    size_t max_seq_len{0};
    infinilm::global_state::FlashMLASchedMeta scheduler_metadata;

    FlashMLAMetadata() = default;

    FlashMLAMetadata(infinicore::Tensor slot_mapping,
                     infinicore::Tensor block_tables,
                     infinicore::Tensor seq_lens,
                     infinicore::Tensor input_offsets,
                     infinicore::Tensor query_start_loc,
                     size_t max_query_len = 0,
                     size_t max_seq_len = 0)
        : slot_mapping(std::move(slot_mapping)),
          block_tables(std::move(block_tables)),
          seq_lens(std::move(seq_lens)),
          input_offsets(std::move(input_offsets)),
          query_start_loc(std::move(query_start_loc)),
          max_query_len(max_query_len),
          max_seq_len(max_seq_len) {}

    bool has_sched_meta() const {
        return slot_mapping && block_tables && seq_lens && scheduler_metadata.has_sched_buffer();
    }
};

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
    std::optional<infinicore::Tensor> extra_topk_length);

namespace backends {

class FlashMLAImpl {
public:
    FlashMLAImpl(size_t num_heads,
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
