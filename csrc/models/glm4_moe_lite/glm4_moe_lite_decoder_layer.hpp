#pragma once

#include "../../config/model_config.hpp"
#include "glm4_moe_lite_attention.hpp"
#include "glm4_moe_lite_moe.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"

#include <memory>
#include <tuple>

namespace infinilm::models::glm4_moe_lite {

class Glm4MoeLiteDecoderLayer : public infinicore::nn::Module {
public:
    Glm4MoeLiteDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(
        const infinicore::Tensor &positions,
        infinicore::Tensor &hidden_states,
        infinicore::Tensor &residual) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    INFINICORE_NN_MODULE(Glm4MoeLiteAttention, self_attn);
    INFINICORE_NN_MODULE(Glm4MoeLiteMLP, mlp);
    INFINICORE_NN_MODULE(Glm4MoeLiteMoE, moe);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
};

} // namespace infinilm::models::glm4_moe_lite
