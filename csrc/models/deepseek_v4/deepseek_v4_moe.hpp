#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v4_gate.hpp"
#include "deepseek_v4_mlp.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/parameter.hpp"
#include "infinicore/tensor.hpp"

#include <infiniccl.h>
#include <memory>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4PackedExperts : public infinicore::nn::Module {
public:
    DeepseekV4PackedExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &topk_indices,
                               const infinicore::Tensor &topk_weights) const;

private:
    INFINICORE_NN_PARAMETER(w13_weight);
    INFINICORE_NN_PARAMETER(w13_weight_scale);
    INFINICORE_NN_PARAMETER(w2_weight);
    INFINICORE_NN_PARAMETER(w2_weight_scale);

    size_t num_experts_{0};
    size_t hidden_size_{0};
    size_t intermediate_size_per_partition_{0};
    int tp_size_{1};
};

class DeepseekV4MoE : public infinicore::nn::Module {
public:
    DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t layer_idx,
                  const infinicore::Device &device);

    infinicore::Tensor forward(infinicore::Tensor hidden_states,
                               const infinicore::Tensor &input_ids) const;
    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    INFINICORE_NN_MODULE(DeepseekV4MoEGate, gate);
    INFINICORE_NN_MODULE(DeepseekV4PackedExperts, experts);
    INFINICORE_NN_MODULE(DeepseekV4MLP, shared_experts);
    int tp_size_{1};
    infinicclComm_t communicator_{nullptr};
};

} // namespace infinilm::models::deepseek_v4
