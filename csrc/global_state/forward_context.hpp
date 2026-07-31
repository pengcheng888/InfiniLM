#pragma once

#include "../models/infinilm_model.hpp"

namespace infinilm::global_state {

struct DeepSeekV4AttentionMetadata {
    infinicore::Tensor swa_indices;
    infinicore::Tensor swa_topk_lengths;

    infinicore::Tensor raw_out_loc;
    infinicore::Tensor page_table;

    infinicore::Tensor c4_out_loc;
    infinicore::Tensor c4_positions;
    infinicore::Tensor c4_topk_lengths_raw;
    infinicore::Tensor c4_sparse_topk_lengths;

    infinicore::Tensor c128_out_loc;
    infinicore::Tensor c128_positions;
    infinicore::Tensor c128_page_indices;
    infinicore::Tensor c128_topk_lengths_clamp1;

    infinicore::Tensor c4_compress_write_loc;
    infinicore::Tensor c4_compress_extra_loc;
    infinicore::Tensor c128_compress_write_loc;

    DeepSeekV4AttentionMetadata() = default;

    explicit DeepSeekV4AttentionMetadata(const infinilm::InfinilmModel::Input &input)
        : swa_indices(input.deepseek_v4.swa_indices),
          swa_topk_lengths(input.deepseek_v4.swa_topk_lengths),
          raw_out_loc(input.deepseek_v4.raw_out_loc),
          page_table(input.deepseek_v4.page_table),
          c4_out_loc(input.deepseek_v4.c4_out_loc),
          c4_positions(input.deepseek_v4.c4_positions),
          c4_topk_lengths_raw(input.deepseek_v4.c4_topk_lengths_raw),
          c4_sparse_topk_lengths(input.deepseek_v4.c4_sparse_topk_lengths),
          c128_out_loc(input.deepseek_v4.c128_out_loc),
          c128_positions(input.deepseek_v4.c128_positions),
          c128_page_indices(input.deepseek_v4.c128_page_indices),
          c128_topk_lengths_clamp1(input.deepseek_v4.c128_topk_lengths_clamp1),
          c4_compress_write_loc(input.deepseek_v4.c4_compress_write_loc),
          c4_compress_extra_loc(input.deepseek_v4.c4_compress_extra_loc),
          c128_compress_write_loc(input.deepseek_v4.c128_compress_write_loc) {}
};

struct DeepSeekV4FlashMLAScheduleCache {
    infinicore::Tensor swa_tile_scheduler_metadata;
    infinicore::Tensor swa_num_splits;
    infinicore::Tensor c4_tile_scheduler_metadata;
    infinicore::Tensor c4_num_splits;
    infinicore::Tensor c128_tile_scheduler_metadata;
    infinicore::Tensor c128_num_splits;

    DeepSeekV4FlashMLAScheduleCache() = default;
};

struct AttentionMetadata {
    /// Past Lengths of cached sequence for each request, of shape `[num_requests]`.
    std::optional<infinicore::Tensor> past_sequence_lengths;
    /// ToTal Lengths for each request sequence, of shape `[num_requests]`.
    std::optional<infinicore::Tensor> total_sequence_lengths;
    /// Offsets of each request in a continous-batched sequence, of shape `[num_requests + 1]`.
    std::optional<infinicore::Tensor> input_offsets;
    /// Cumulative total sequence lengths for each request, of shape `[num_requests + 1]`.
    std::optional<infinicore::Tensor> cu_seqlens;
    /// Block ids for each request `[batch, max_block_table_length]`. Used for paged cache.
    std::optional<infinicore::Tensor> block_tables;
    /// Slot ids for each token `[seq]`. Used for paged cache.
    std::optional<infinicore::Tensor> slot_mapping;

    AttentionMetadata() = default;

    AttentionMetadata(std::optional<infinicore::Tensor> past_sequence_lengths,
                      std::optional<infinicore::Tensor> total_sequence_lengths,
                      std::optional<infinicore::Tensor> input_offsets,
                      std::optional<infinicore::Tensor> cu_seqlens,
                      std::optional<infinicore::Tensor> block_tables,
                      std::optional<infinicore::Tensor> slot_mapping) : past_sequence_lengths(past_sequence_lengths),
                                                                        total_sequence_lengths(total_sequence_lengths),
                                                                        input_offsets(input_offsets),
                                                                        cu_seqlens(cu_seqlens),
                                                                        block_tables(block_tables),
                                                                        slot_mapping(slot_mapping) {}

    explicit AttentionMetadata(const infinilm::InfinilmModel::Input &input) : AttentionMetadata(input.past_sequence_lengths,
                                                                                                input.total_sequence_lengths,
                                                                                                input.input_offsets,
                                                                                                input.cu_seqlens,
                                                                                                input.block_tables,
                                                                                                input.slot_mapping) {}
};

struct DeepSeekV4LayerKVCache {
    infinicore::Tensor swa_cache_raw;
    infinicore::Tensor c4_cache_raw;
    infinicore::Tensor c4_indexer_cache_raw;
    infinicore::Tensor c128_cache_raw;
    infinicore::Tensor kv_scale;
    infinicore::Tensor compressor_state;
    infinicore::Tensor indexer_compressor_state;
};

struct MultiModalMetadata {
    std::optional<std::vector<size_t>> image_req_ids;
    // Flattened [start, end) token ranges in the current packed language sequence.
    std::optional<std::vector<size_t>> visual_token_ranges;
};

struct MambaMetadata {
    /// Offsets of each request in a continous-batched sequence, of shape `[num_requests + 1]`.
    std::optional<infinicore::Tensor> input_offsets;
    /// State cache indices read at the start of each request forward.
    std::optional<infinicore::Tensor> init_state_indices;
    /// State cache indices written with the final state of each request forward.
    std::optional<infinicore::Tensor> final_state_indices;
};

struct ForwardContext {
    AttentionMetadata attn_metadata;
    DeepSeekV4AttentionMetadata deepseek_v4_attention_metadata;
    DeepSeekV4FlashMLAScheduleCache deepseek_v4_flashmla_schedule_cache;
    MambaMetadata mamba_metadata;
    MultiModalMetadata mm_metadata;
    std::vector<infinicore::Tensor> kv_cache_vec;
    std::vector<DeepSeekV4LayerKVCache> deepseek_v4_kv_cache_vec;
    std::vector<infinicore::Tensor> conv_state_vec;
    std::vector<infinicore::Tensor> ssm_state_vec;
};

void initialize_forward_context(ForwardContext &forward_context);

ForwardContext &get_forward_context();

} // namespace infinilm::global_state
