#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::glm4_moe_lite {

class Glm4MoeLiteMLP : public infinicore::nn::Module {
public:
    Glm4MoeLiteMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   const infinicore::Device &device);

    infinicore::Tensor forward(infinicore::Tensor hidden_states) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;

    size_t intermediate_size_per_partition_{0};
};

} // namespace infinilm::models::glm4_moe_lite
