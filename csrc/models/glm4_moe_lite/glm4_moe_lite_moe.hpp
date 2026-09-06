#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "../../layers/moe/router/topk_router.hpp"
#include "../qwen3_moe/qwen3_moe_experts.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::glm4_moe_lite {

using Glm4MoeLiteMLP = infinilm::layers::mlp::MLP;

class Glm4MoeLiteMoE : public infinicore::nn::Module {
public:
    Glm4MoeLiteMoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   size_t layer_idx,
                   const infinicore::Device &device);

    infinicore::Tensor forward(infinicore::Tensor hidden_states) const;
    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    INFINICORE_NN_MODULE(infinilm::layers::moe::TopKRouter, gate);
    INFINICORE_NN_MODULE(infinilm::models::qwen3_moe::Qwen3MoeExperts, experts);
    INFINICORE_NN_MODULE(Glm4MoeLiteMLP, shared_experts);
    float routed_scaling_factor_{1.0f};
};

} // namespace infinilm::models::glm4_moe_lite
