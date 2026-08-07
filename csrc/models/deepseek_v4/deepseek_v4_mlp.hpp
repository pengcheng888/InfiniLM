#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "deepseek_v4_scratch.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4MLP : public infinicore::nn::Module {
public:
    DeepseekV4MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device);

    infinicore::Tensor forward(infinicore::Tensor hidden_states) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> w2_;
    static thread_local DeepseekV4SharedExpertScratch shared_scratch_;
    size_t intermediate_size_per_partition_{0};
    infinicore::DataType dtype_{infinicore::DataType::BF16};
    infinicore::Device device_;
};

} // namespace infinilm::models::deepseek_v4
