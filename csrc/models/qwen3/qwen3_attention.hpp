#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../global_state/forward_context.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/linear/linear.hpp"
#include "qwen3_rms_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <tuple>

namespace infinilm::models::qwen3 {

class Qwen3Attention : public infinicore::nn::Module {
public:
    Qwen3Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   size_t layer_idx,
                   const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    void process_weights_after_loading() override {
        qkv_proj_->process_weights_after_loading();
    }

    void reset_runtime_state() const override {
        qkv_proj_->reset_runtime_state();
    }

private:
    infinicore::Tensor forward_paged_(const infinicore::Tensor &positions,
                                      const infinicore::Tensor &hidden_states) const;

    infinicore::Tensor prepare_position_ids_(const infinicore::Tensor &position_ids,
                                             size_t seq_len) const;

    std::tuple<infinicore::Tensor, infinicore::Tensor> do_kv_cache_update(
        const infinicore::Tensor key,
        const infinicore::Tensor value,
        infinicore::Tensor &kv_cache,
        const infinicore::Tensor slot_mapping) const;

    infinicore::Tensor caculate_attention(
        const infinicore::Tensor &query,
        const infinicore::Tensor &key_cache,
        const infinicore::Tensor &value_cache,
        const infinilm::global_state::AttentionMetadata &attn_metadata) const;

    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    INFINICORE_NN_MODULE(Qwen3RMSNorm, q_norm);
    INFINICORE_NN_MODULE(Qwen3RMSNorm, k_norm);

    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t hidden_size_;
    size_t head_dim_;
    size_t q_size_;
    size_t k_size_;
    size_t v_size_;
    size_t max_position_embeddings_;
    size_t rotary_dim_;
    float rms_norm_eps_;
    float rope_theta_;
    float rope_factor_;
    float rope_low_;
    float rope_high_;
    float rope_attention_factor_;
    float scale_;
    bool is_neox_;
    ::infinilm::backends::AttentionBackend attention_backend_;

    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);
};

} // namespace infinilm::models::qwen3
