#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v4_attention.hpp"
#include "deepseek_v4_moe.hpp"
#include "deepseek_v4_scratch.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <tuple>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4DecoderLayer : public infinicore::nn::Module {
public:
    DeepseekV4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device);

    infinicore::Tensor forward_naive(const infinicore::Tensor &positions,
                                     infinicore::Tensor &hidden_states,
                                     const infinicore::Tensor &input_ids);

    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
    forward(const infinicore::Tensor &positions,
            infinicore::Tensor &hidden_states,
            const infinicore::Tensor &input_ids,
            const infinicore::Tensor &prev_residual,
            const infinicore::Tensor &prev_post,
            const infinicore::Tensor &prev_comb);

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    infinicore::Tensor complete_deferred_hc_post(const infinicore::Tensor &hidden_states,
                                                 const infinicore::Tensor &residual,
                                                 const infinicore::Tensor &post,
                                                 const infinicore::Tensor &comb) const;

    INFINICORE_NN_MODULE(DeepseekV4Attention, attn);
    INFINICORE_NN_MODULE(DeepseekV4MoE, ffn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, attn_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, ffn_norm);

    INFINICORE_NN_PARAMETER(hc_attn_fn);
    INFINICORE_NN_PARAMETER(hc_ffn_fn);
    INFINICORE_NN_PARAMETER(hc_attn_base);
    INFINICORE_NN_PARAMETER(hc_ffn_base);
    INFINICORE_NN_PARAMETER(hc_attn_scale);
    INFINICORE_NN_PARAMETER(hc_ffn_scale);

    size_t layer_idx_;
    infinicore::DataType dtype_{infinicore::DataType::BF16};
    infinicore::Device device_;
    size_t hidden_size_{0};
    size_t hc_mult_{4};
    double rms_norm_eps_{1e-6};
    double hc_eps_{1e-6};
    int hc_sinkhorn_iters_{20};
    size_t compress_ratio_{0};
    bool is_last_layer_{false};
    bool debug_dump_enabled_{false};
    bool use_fused_mhc_post_pre_{true};
    static thread_local DeepseekV4DecoderLayerSharedScratch shared_scratch_;
};

} // namespace infinilm::models::deepseek_v4
