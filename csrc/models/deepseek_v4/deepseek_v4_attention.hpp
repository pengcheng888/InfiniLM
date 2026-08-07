#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/linear/linear.hpp"
#include "deepseek_v4_c4_indexer.hpp"
#include "deepseek_v4_compressor.hpp"
#include "deepseek_v4_rms_norm.hpp"
#include "deepseek_v4_scratch.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <optional>

namespace infinilm::global_state {
struct DSV4AttnMetadata;
struct FlashMLASchedMeta;
struct DeepSeekV4LayerKVCache;
struct ForwardContext;
} // namespace infinilm::global_state

namespace infinicore::op {
struct DeepseekV4FlashMLASparseAttentionSchedule;
} // namespace infinicore::op

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Attention : public infinicore::nn::Module {
public:
    DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        size_t layer_idx,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    size_t compress_ratio() const {
        return compress_ratio_;
    }

    void process_weights_after_loading() override {
        wqkv_a_->process_weights_after_loading();
        wq_b_->process_weights_after_loading();
        wo_a_->process_weights_after_loading();
        wo_b_->process_weights_after_loading();
        if (compressor_) {
            compressor_->process_weights_after_loading();
        }
        if (indexer_) {
            indexer_->process_weights_after_loading();
        }
        attention_scratch_.preallocate_attn_out(num_local_attention_heads_, head_dim_, dtype_, device_);
    }

    void reset_runtime_state() const override {
        wqkv_a_->reset_runtime_state();
        wq_b_->reset_runtime_state();
        wo_a_->reset_runtime_state();
        wo_b_->reset_runtime_state();
        if (compressor_) {
            compressor_->reset_runtime_state();
        }
        if (indexer_) {
            indexer_->reset_runtime_state();
        }
    }

private:
    struct ForwardPrepareResult {
        infinicore::Tensor q;
        std::optional<infinicore::Tensor> extra_raw_cache;
        std::optional<infinicore::Tensor> extra_indices;
        std::optional<infinicore::Tensor> extra_topk_lengths;
        int extra_page_size{0};
    };

    ForwardPrepareResult _forward_prepare(const infinicore::Tensor &hidden_states,
                                          const infinicore::Tensor &pos_ids,
                                          size_t seq_len,
                                          infinilm::global_state::DSV4AttnMetadata &dsv4_metadata,
                                          infinilm::global_state::DeepSeekV4LayerKVCache &layer_cache) const;

    infinicore::Tensor _compute_q_b_and_kv(const infinicore::Tensor &q_lora,
                                           infinicore::Tensor &kv,
                                           const infinicore::Tensor &pos_ids,
                                           size_t seq_len) const;

    infinicore::Tensor _compute_fused_q_b_and_kv(const infinicore::Tensor &q_lora,
                                                 infinicore::Tensor &kv,
                                                 const infinicore::Tensor &pos_ids,
                                                 size_t seq_len) const;

    void apply_rope_(const infinicore::Tensor &positions,
                     infinicore::Tensor query,
                     std::optional<infinicore::Tensor> key,
                     bool inverse) const;
    infinicore::Tensor prepare_attn_out_workspace(size_t seq_len,
                                                  infinicore::DataType dtype,
                                                  const infinicore::Device &device) const;
    void cache_flashmla_schedule_metadata(
        infinilm::global_state::FlashMLASchedMeta &flashmla_metadata,
        const infinicore::op::DeepseekV4FlashMLASparseAttentionSchedule &flashmla_schedule) const;
    void refresh_flashmla_schedule_metadata(
        infinilm::global_state::FlashMLASchedMeta &flashmla_metadata,
        const infinicore::Tensor &indices,
        const infinicore::Tensor &topk_lengths,
        std::optional<infinicore::Tensor> extra_indices,
        std::optional<infinicore::Tensor> extra_topk_lengths) const;
    void compute_sparse_attention(
        infinicore::Tensor attn_out,
        const infinicore::Tensor &q,
        size_t seq_len,
        const infinicore::Device &device,
        const infinicore::Tensor &swa_cache_raw,
        const infinicore::Tensor &swa_indices,
        const infinicore::Tensor &swa_topk_lengths,
        std::optional<infinicore::Tensor> extra_raw_cache,
        std::optional<infinicore::Tensor> extra_indices,
        std::optional<infinicore::Tensor> extra_topk_lengths,
        int extra_page_size,
        infinilm::global_state::DSV4AttnMetadata &dsv4_metadata) const;
    void validate_forward_metadata_and_cache(
        const infinilm::global_state::ForwardContext &forward_context) const;

    std::shared_ptr<infinilm::layers::linear::FusedReplicatedLinear> wqkv_a_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wq_b_;
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, q_norm);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, kv_norm);
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wo_a_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> wo_b_;
    std::shared_ptr<DeepseekV4Compressor> compressor_;
    INFINICORE_NN_MODULE(DeepseekV4C4Indexer, indexer);
    INFINICORE_NN_PARAMETER(attn_sink);

    infinicore::DataType dtype_;
    infinicore::Device device_;
    infinicore::Tensor rope_freqs_cis_;
    infinicore::Tensor attn_sink_for_flash_;
    static thread_local DeepseekV4AttentionScratch attention_scratch_;
    static thread_local DeepseekV4MLAScratch mla_scratch_;

    size_t layer_idx_;
    size_t hidden_size_;
    size_t head_dim_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t q_lora_rank_;
    size_t o_lora_rank_;
    size_t qk_rope_head_dim_;
    size_t o_groups_;
    size_t num_local_attention_heads_{0};
    size_t num_local_groups_{0};
    size_t tp_rank_{0};
    size_t tp_size_{1};
    size_t compress_ratio_{0};
    mutable bool fused_q_b_and_kv_{true};
    float flashmla_softmax_scale_{1.0f};
    size_t max_position_embeddings_{0};
    double rope_theta_{10000.0};
    double compress_rope_theta_{160000.0};
    double rope_factor_{1.0};
    double rope_beta_fast_{32.0};
    double rope_beta_slow_{1.0};
    size_t rope_original_seq_len_{0};
    double rms_norm_eps_;
};

} // namespace infinilm::models::deepseek_v4
