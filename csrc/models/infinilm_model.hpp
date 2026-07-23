#pragma once

#include "../backends/attention_backends.hpp"
#include "../cache/cache.hpp"
#include "../config/model_config.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <optional>
#include <vector>

namespace infinilm {
class InfinilmModel : public infinicore::nn::Module {
public:
    struct Config {
        std::string model_type;
        virtual ~Config() = default;
    };

    struct Input {
        /// Token IDs tensor of shape `[batch, seq_len]`.
        std::optional<infinicore::Tensor> input_ids;
        /// Position IDs tensor of shape `[batch, seq_len]` or `[seq_len]`.
        std::optional<infinicore::Tensor> position_ids;
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
        /// DeepSeek-V4 SWA FlashMLA indices, shape `[seq, 128]`.
        std::optional<infinicore::Tensor> dsv4_swa_indices;
        /// DeepSeek-V4 SWA top-k lengths, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_swa_topk_lengths;
        /// DeepSeek-V4 C4 compressed FlashMLA indices, shape `[seq, 512]`.
        std::optional<infinicore::Tensor> dsv4_c4_indices;
        /// DeepSeek-V4 C4 compressed top-k lengths, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c4_topk_lengths;
        /// DeepSeek-V4 C128 compressed FlashMLA indices, shape `[seq, 64]`.
        std::optional<infinicore::Tensor> dsv4_c128_indices;
        /// DeepSeek-V4 C128 compressed top-k lengths, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c128_topk_lengths;
        /// DeepSeek-V4 raw output cache locations, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_raw_out_loc;
        /// DeepSeek-V4 repeated page table, shape `[seq, max_pages]`.
        std::optional<infinicore::Tensor> dsv4_page_table;
        /// DeepSeek-V4 causal sequence lengths per query token, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_seq_lens_casual;
        /// DeepSeek-V4 raw positions per query token, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_positions_casual;
        /// DeepSeek-V4 C4 compressed output locations, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c4_out_loc;
        /// DeepSeek-V4 C4 compression RoPE positions, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c4_positions;
        /// DeepSeek-V4 C4 raw compressed lengths, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c4_topk_lengths_raw;
        /// DeepSeek-V4 C4 compressed lengths clamped to at least 1, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c4_topk_lengths_clamp1;
        /// DeepSeek-V4 C4 sparse page indices, shape `[seq, 512]`.
        std::optional<infinicore::Tensor> dsv4_c4_sparse_indices;
        /// DeepSeek-V4 C4 sparse top-k lengths, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c4_sparse_topk_lengths;
        /// DeepSeek-V4 C128 compressed output locations, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c128_out_loc;
        /// DeepSeek-V4 C128 compression RoPE positions, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c128_positions;
        /// DeepSeek-V4 C128 page indices, shape `[seq, 64]` in the current adapter.
        std::optional<infinicore::Tensor> dsv4_c128_page_indices;
        /// DeepSeek-V4 C128 top-k lengths clamped to at least 1, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c128_topk_lengths_clamp1;
        /// DeepSeek-V4 C4 compressor write locations, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c4_compress_write_loc;
        /// DeepSeek-V4 C4 compressor overlap locations, shape `[seq, 1]`.
        std::optional<infinicore::Tensor> dsv4_c4_compress_extra_loc;
        /// DeepSeek-V4 C4 compressor state indices, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c4_compress_state_indices;
        /// DeepSeek-V4 C128 compressor write locations, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c128_compress_write_loc;
        /// DeepSeek-V4 C128 compressor state indices, shape `[seq]`.
        std::optional<infinicore::Tensor> dsv4_c128_compress_state_indices;
        /// Mamba state cache indices read at the start of each request forward, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> mamba_init_state_indices;
        /// Mamba state cache indices written with the final state of each request forward, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> mamba_final_state_indices;
        /// Image pixel values for multi-modal models.
        /// Vector of tensors. Shape is model-specific (e.g. LLaVA: [batch, 3, H, W], MiniCPM-V: [n_patch, 3, filter_H, H * W / filter_H]).
        std::optional<std::vector<infinicore::Tensor>> pixel_values;
        /// Image placeholder bounds for MiniCPM-V style replacement.
        /// Vector of tensors shape: [n_patch, 2].
        std::optional<std::vector<infinicore::Tensor>> image_bound;
        /// Target patch sizes for each image (MiniCPM-V).
        /// Vector of tensors shape: [n_path, 2] if pre-flattened.
        std::optional<std::vector<infinicore::Tensor>> tgt_sizes;
        /// Qwen-style image grids. Vector of tensors shape: [3] with temporal, height, width.
        std::optional<std::vector<infinicore::Tensor>> image_grid_thw;
        /// req_id for each pixel_values among a batch.
        std::optional<std::vector<size_t>> image_req_ids;
        /// Flattened [start, end) visual token ranges in the packed language sequence.
        std::optional<std::vector<size_t>> visual_token_ranges;
        /// Target model hidden states consumed by draft/MTP models.
        std::optional<infinicore::Tensor> target_hidden_states;
    };

    struct Output {
        /// Logits.
        infinicore::Tensor logits;
        /// Optional final hidden states, used by MTP/Eagle draft models.
        infinicore::Tensor hidden_states;
    };

    virtual ~InfinilmModel() = default;
    virtual Output forward(const Input &input) const = 0;
    virtual void reset_cache(const cache::CacheConfig *cache_config);
    virtual const cache::CacheConfig *get_cache_config() const {
        return cache_config_.get();
    }

    void process_weights_after_loading();
    void reset_runtime_state() const;

protected:
    std::vector<infinicore::Tensor> default_allocate_kv_cache_tensors(
        const cache::CacheConfig *cache_config,
        const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
        const backends::AttentionBackend &attention_backend);

    std::unique_ptr<cache::CacheConfig> cache_config_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;

private:
    static void process_weights_recursive_(infinicore::nn::Module *module);
    static void reset_runtime_state_recursive_(const infinicore::nn::Module *module);
};
} // namespace infinilm
