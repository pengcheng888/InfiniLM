#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/mla_attention/backends/flashmla.hpp"
#include "deepseek_v4_rope.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <optional>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Attention : public infinicore::nn::Module {
public:
    DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        size_t layer_idx,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    infinicore::Tensor compute_sparse_attention(const infinicore::Tensor &q,
                                                const infinicore::Tensor &kv,
                                                size_t tokens) const;
    infinicore::Tensor apply_output_projection_(const infinicore::Tensor &attn_out,
                                                size_t tokens) const;
    infinicore::Tensor raw_cache_view_(const infinicore::Tensor &raw_cache) const;
    void apply_rope_(const infinicore::Tensor &positions,
                     infinicore::Tensor query,
                     std::optional<infinicore::Tensor> key,
                     bool inverse) const;

    std::shared_ptr<infinilm::layers::linear::FusedReplicatedLinear> wqkv_a_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wq_b_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, kv_norm);
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> wo_a_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> wo_b_;
    std::shared_ptr<infinilm::layers::mla_attention::backends::SparseFlashMLAImpl> sparse_mla_attn_;
    infinicore::Tensor rope_freqs_cis_;
    INFINICORE_NN_PARAMETER(attn_sink);

    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t head_dim_{0};
    size_t qk_rope_head_dim_{0};
    size_t num_attention_heads_{0};
    size_t num_local_attention_heads_{0};
    size_t num_local_groups_{0};
    size_t q_lora_rank_{0};
    size_t o_lora_rank_{0};
    size_t o_groups_{1};
    size_t compress_ratio_{0};
    size_t max_position_embeddings_{0};
    size_t tp_rank_{0};
    size_t tp_size_{1};
    float softmax_scale_{1.0f};
    double rms_norm_eps_{1e-6};
    double rope_theta_{10000.0};
    double compress_rope_theta_{160000.0};
    double rope_factor_{1.0};
    double rope_beta_fast_{32.0};
    double rope_beta_slow_{1.0};
    size_t rope_original_seq_len_{0};
    infinicore::DataType dtype_;
    infinicore::Device device_;
};

} // namespace infinilm::models::deepseek_v4
