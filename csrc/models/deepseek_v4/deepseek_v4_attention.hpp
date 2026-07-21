#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "deepseek_v4_rms_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
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

private:
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wq_b_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> weights_proj_;
    INFINICORE_NN_MODULE(DeepseekV4Compressor, compressor);
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
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wq_a_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wkv_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wq_b_;
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, q_norm);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, kv_norm);
    std::shared_ptr<DeepseekV4RMSNorm> wq_b_post_norm_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wo_a_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> wo_b_;
    INFINICORE_NN_MODULE(DeepseekV4Compressor, compressor);
    INFINICORE_NN_MODULE(DeepseekV4C4Indexer, indexer);
    INFINICORE_NN_PARAMETER(attn_sink);

    size_t layer_idx_;
    size_t hidden_size_;
    size_t head_dim_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t q_lora_rank_;
    size_t o_lora_rank_;
    size_t qk_rope_head_dim_;
    size_t o_groups_;
    double rms_norm_eps_;
};

} // namespace infinilm::models::deepseek_v4
