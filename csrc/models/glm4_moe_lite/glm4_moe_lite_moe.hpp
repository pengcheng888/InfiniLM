#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/moe/router/topk_router.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::glm4_moe_lite {

class Glm4MoeLiteSharedMLP : public infinicore::nn::Module {
public:
    Glm4MoeLiteSharedMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device);

    infinicore::Tensor forward(infinicore::Tensor hidden_states) const;
    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;
    size_t intermediate_size_per_partition_{0};
};

class Glm4MoeLiteExperts : public infinicore::nn::Module {
public:
    Glm4MoeLiteExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &topk_indices,
                               const infinicore::Tensor &topk_weights) const;

private:
    size_t num_experts_{0};
    size_t intermediate_size_per_partition_{0};
    infinicclComm_t communicator_{nullptr};
    size_t tp_size_{1};

    std::vector<infinicore::Tensor> gate_weights_;
    std::vector<infinicore::Tensor> up_weights_;
    std::vector<infinicore::Tensor> down_weights_;
};

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
    INFINICORE_NN_MODULE(Glm4MoeLiteExperts, experts);
    INFINICORE_NN_MODULE(Glm4MoeLiteSharedMLP, shared_experts);
};

} // namespace infinilm::models::glm4_moe_lite
