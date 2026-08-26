#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/mla_attention/backends/mla_attention_layer.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::glm4_moe_lite {

class Glm4MoeLiteAttention : public infinicore::nn::Module {
public:
    Glm4MoeLiteAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    void apply_rope_(const infinicore::Tensor &positions,
                     infinicore::Tensor query,
                     infinicore::Tensor key) const;

    infinicore::Tensor forward_mha(const infinicore::Tensor &q,
                                   const infinicore::Tensor &kv_c,
                                   const infinicore::Tensor &k_pe,
                                   size_t tokens) const;

    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t num_attention_heads_{0};
    size_t num_local_attention_heads_{0};
    size_t q_lora_rank_{0};
    size_t kv_lora_rank_{0};
    size_t qk_nope_head_dim_{0};
    size_t qk_rope_head_dim_{0};
    size_t qk_head_dim_{0};
    size_t v_head_dim_{0};
    size_t tp_rank_{0};
    size_t tp_size_{1};
    double rms_norm_eps_{1e-6};
    double softmax_scale_{1.0};

    infinicore::Tensor rope_freqs_cis_;
    infinicore::Tensor w_kc_;
    infinicore::Tensor w_kc_t_;
    infinicore::Tensor w_vc_;

    std::shared_ptr<infinilm::layers::linear::FusedReplicatedLinear> qkv_a_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> q_b_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> kv_b_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    std::shared_ptr<infinilm::layers::mla_attention::MLAAttentionLayer> mla_attn_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_a_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, kv_a_layernorm);
};

} // namespace infinilm::models::glm4_moe_lite
