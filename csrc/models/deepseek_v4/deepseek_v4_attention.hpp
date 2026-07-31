#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "deepseek_v4_compressor.hpp"
#include "deepseek_v4_rms_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <optional>
#include <utility>

namespace infinilm::global_state {
struct DeepSeekV4AttentionMetadata;
struct DeepSeekV4FlashMLAScheduleCache;
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

    void process_weights_after_loading() override {
        wq_a_->process_weights_after_loading();
        wkv_->process_weights_after_loading();
        wq_b_->process_weights_after_loading();
        wo_a_->process_weights_after_loading();
        wo_b_->process_weights_after_loading();
        if (csa_compressor_) {
            csa_compressor_->process_weights_after_loading();
        }
        if (indexer_) {
            indexer_->process_weights_after_loading();
        }
        if (hca_compressor_) {
            hca_compressor_->process_weights_after_loading();
        }
    }

    void reset_runtime_state() const override {
        wq_a_->reset_runtime_state();
        wkv_->reset_runtime_state();
        wq_b_->reset_runtime_state();
        wo_a_->reset_runtime_state();
        wo_b_->reset_runtime_state();
        if (csa_compressor_) {
            csa_compressor_->reset_runtime_state();
        }
        if (indexer_) {
            indexer_->reset_runtime_state();
        }
        if (hca_compressor_) {
            hca_compressor_->reset_runtime_state();
        }
    }

private:
    void apply_rope_(const infinicore::Tensor &positions,
                     infinicore::Tensor query,
                     std::optional<infinicore::Tensor> key,
                     bool inverse) const;
    infinicore::Tensor prepare_attn_out_workspace(size_t seq_len,
                                                  infinicore::DataType dtype,
                                                  const infinicore::Device &device) const;
    std::pair<std::optional<infinicore::Tensor>, std::optional<infinicore::Tensor>>
    prepare_flashmla_schedule_metadata(
        const infinilm::global_state::DeepSeekV4FlashMLAScheduleCache &schedule_cache) const;
    void cache_flashmla_schedule_metadata(
        infinilm::global_state::DeepSeekV4FlashMLAScheduleCache &schedule_cache,
        const infinicore::op::DeepseekV4FlashMLASparseAttentionSchedule &flashmla_schedule) const;
    void validate_forward_metadata_and_cache(
        const infinilm::global_state::ForwardContext &forward_context) const;

    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wq_a_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wkv_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wq_b_;
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, q_norm);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, kv_norm);
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wo_a_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> wo_b_;
    std::shared_ptr<DeepseekV4CSACompressor> csa_compressor_;
    INFINICORE_NN_MODULE(DeepseekV4C4Indexer, indexer);
    std::shared_ptr<DeepseekV4HCACompressor> hca_compressor_;
    INFINICORE_NN_PARAMETER(attn_sink);

    infinicore::DataType dtype_;
    infinicore::Tensor rope_freqs_cis_;
    infinicore::Tensor attn_sink_for_flash_;
    mutable infinicore::Tensor attn_out_workspace_;

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
