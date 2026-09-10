#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v4_attention.hpp"
#include "deepseek_v4_moe.hpp"
#include "deepseek_v4_scratch.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4DecoderLayer : public infinicore::nn::Module {
public:
    DeepseekV4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &input_ids) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    INFINICORE_NN_MODULE(DeepseekV4Attention, self_attn);
    INFINICORE_NN_MODULE(DeepseekV4MoE, moe);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);

    INFINICORE_NN_PARAMETER(hc_attn_fn);
    INFINICORE_NN_PARAMETER(hc_ffn_fn);
    INFINICORE_NN_PARAMETER(hc_attn_base);
    INFINICORE_NN_PARAMETER(hc_ffn_base);
    INFINICORE_NN_PARAMETER(hc_attn_scale);
    INFINICORE_NN_PARAMETER(hc_ffn_scale);

    size_t layer_idx_{0};
    infinicore::DataType dtype_{infinicore::DataType::BF16};
    infinicore::Device device_;
    size_t hidden_size_{0};
    size_t hc_mult_{4};
    double rms_norm_eps_{1e-6};
    double hc_eps_{1e-6};
    int hc_sinkhorn_iters_{20};
    static thread_local DeepseekV4DecoderLayerSharedScratch shared_scratch_;
};

} // namespace infinilm::models::deepseek_v4
