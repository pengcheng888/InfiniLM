#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "deepseek_v4_rms_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Compressor : public infinicore::nn::Module {
public:
    DeepseekV4Compressor(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t compress_ratio,
                         size_t compressor_head_dim,
                         const infinicore::Device &device);

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

    infinicore::Tensor forward_kv_score(const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor ape() const { return ape_; }
    infinicore::Tensor norm_weight() const { return norm_->weight(); }
    float norm_eps() const { return norm_->eps(); }

private:
    INFINICORE_NN_PARAMETER(ape);
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wgate_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wkv_;
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, norm);
};

class DeepseekV4C4Indexer : public infinicore::nn::Module {
public:
    DeepseekV4C4Indexer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        const infinicore::Device &device);

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

    infinicore::Tensor compute_q(const infinicore::Tensor &q_lora, size_t seq_len) const;
    infinicore::Tensor compute_weights(const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor forward_kv_score(const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor ape() const { return compressor_->ape(); }
    infinicore::Tensor norm_weight() const { return compressor_->norm_weight(); }
    float norm_eps() const { return compressor_->norm_eps(); }
    float weight_scale() const { return weight_scale_; }
    size_t index_n_heads() const { return index_n_heads_; }
    size_t index_head_dim() const { return index_head_dim_; }

private:
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wq_b_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> weights_proj_;
    INFINICORE_NN_MODULE(DeepseekV4Compressor, compressor);
    size_t index_n_heads_{0};
    size_t index_head_dim_{0};
    float weight_scale_{1.0f};
};

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
        if (compressor_) {
            compressor_->process_weights_after_loading();
        }
        if (indexer_) {
            indexer_->process_weights_after_loading();
        }
    }

    void reset_runtime_state() const override {
        wq_a_->reset_runtime_state();
        wkv_->reset_runtime_state();
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
    infinicore::Tensor build_rope_freqs_cis(const infinicore::Device &device) const;
    void apply_rope_(const infinicore::Tensor &positions,
                     infinicore::Tensor query,
                     std::optional<infinicore::Tensor> key,
                     bool inverse) const;
    infinicore::Tensor attn_out_workspace(size_t seq_len,
                                          infinicore::DataType dtype,
                                          const infinicore::Device &device) const;

    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wq_a_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wkv_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wq_b_;
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, q_norm);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, kv_norm);
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wo_a_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> wo_b_;
    INFINICORE_NN_MODULE(DeepseekV4Compressor, compressor);
    INFINICORE_NN_MODULE(DeepseekV4C4Indexer, indexer);
    INFINICORE_NN_PARAMETER(attn_sink);

    infinicore::DataType dtype_;
    infinicore::Tensor rope_freqs_cis_;
    mutable infinicore::Tensor attn_out_workspace_;

    size_t layer_idx_;
    size_t hidden_size_;
    size_t head_dim_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t q_lora_rank_;
    size_t o_lora_rank_;
    size_t qk_rope_head_dim_;
    size_t index_head_dim_;
    size_t o_groups_;
    size_t num_local_attention_heads_{0};
    size_t num_local_groups_{0};
    size_t tp_rank_{0};
    size_t tp_size_{1};
    size_t compress_ratio_{0};
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
