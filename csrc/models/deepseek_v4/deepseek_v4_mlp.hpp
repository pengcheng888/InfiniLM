#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/module.hpp"

#include <memory>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4MLP : public infinicore::nn::Module {
public:
    DeepseekV4MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;
    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> w2_;
};

} // namespace infinilm::models::deepseek_v4
