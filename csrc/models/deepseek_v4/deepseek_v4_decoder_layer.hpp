#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v4_attention.hpp"
#include "deepseek_v4_moe.hpp"
#include "deepseek_v4_rms_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <tuple>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4DecoderLayer : public infinicore::nn::Module {
public:
    DeepseekV4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual,
                                                               const infinicore::Tensor &input_ids);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &input_ids);

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    INFINICORE_NN_MODULE(DeepseekV4Attention, attn);
    INFINICORE_NN_MODULE(DeepseekV4MoE, ffn);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, attn_norm);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, ffn_norm);

    INFINICORE_NN_PARAMETER(hc_attn_fn);
    INFINICORE_NN_PARAMETER(hc_ffn_fn);
    INFINICORE_NN_PARAMETER(hc_attn_base);
    INFINICORE_NN_PARAMETER(hc_ffn_base);
    INFINICORE_NN_PARAMETER(hc_attn_scale);
    INFINICORE_NN_PARAMETER(hc_ffn_scale);

    size_t layer_idx_;
    size_t hidden_size_{0};
    size_t hc_mult_{4};
    double rms_norm_eps_{1e-6};
    double hc_eps_{1e-6};
    int hc_sinkhorn_iters_{20};
    bool mhc_pre_kernel_backend_enabled_{false};
    bool mhc_post_kernel_backend_enabled_{false};
    bool debug_dump_enabled_{false};
};

} // namespace infinilm::models::deepseek_v4
