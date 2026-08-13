#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v4_gate.hpp"
#include "deepseek_v4_mlp.hpp"
#include "deepseek_v4_scratch.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include "moe_backends/routed_expert_backend.hpp"

#include <memory>
#include <optional>
#include <string>
#include <tuple>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4PackedExperts : public infinicore::nn::Module {
public:
    DeepseekV4PackedExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            const infinicore::Device &device);

    infinicore::Tensor w13_weight() const { return w13_weight_; }
    infinicore::Tensor w13_weight_scale() const { return w13_weight_scale_; }
    infinicore::Tensor w2_weight() const { return w2_weight_; }
    infinicore::Tensor w2_weight_scale() const { return w2_weight_scale_; }
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &topk_weights,
                               const infinicore::Tensor &topk_indices,
                               const std::optional<infinicore::Tensor> &shared_output = std::nullopt) const;

    void process_weights_after_loading() override;

private:
    moe_backends::RoutedExpertContext make_backend_context() const;

    INFINICORE_NN_PARAMETER(w13_weight);
    INFINICORE_NN_PARAMETER(w13_weight_scale);
    INFINICORE_NN_PARAMETER(w2_weight);
    INFINICORE_NN_PARAMETER(w2_weight_scale);

    size_t num_experts_{0};
    size_t hidden_size_{0};
    size_t intermediate_size_{0};
    size_t intermediate_size_per_partition_{0};
    size_t num_experts_per_tok_{0};
    double routed_scaling_factor_{1.0};
    moe_backends::RoutedExpertBackendChoice routed_expert_backend_;
    int marlin_block_size_{16};
    int marlin_mode_{54};
    int marlin_delta_{1};
    moe_backends::MarlinGemmOverride marlin_gemm_override_;
    mutable infinicore::Tensor w13_weight_marlin_;
    mutable infinicore::Tensor w2_weight_marlin_;
    static thread_local DeepseekV4RoutedExpertScratch shared_scratch_;
    infinicore::DataType dtype_{infinicore::DataType::BF16};
    infinicore::Device device_;
    bool marlin_only_weights_{false};
};

class DeepseekV4MoE : public infinicore::nn::Module {
public:
    DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t layer_idx,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &input_ids) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    INFINICORE_NN_MODULE(DeepseekV4MoEGate, gate);
    INFINICORE_NN_MODULE(DeepseekV4PackedExperts, experts);
    INFINICORE_NN_MODULE(DeepseekV4PackedMLP, shared_experts);

    size_t layer_idx_{0};
    size_t tp_size_{1};
    int tp_rank_{0};
    infinicclComm_t communicator_{nullptr};
    bool debug_dump_enabled_{false};
    bool moe_allreduce_outplace_enabled_{false};
    bool moe_custom_allreduce_enabled_{false};
    mutable DeepseekV4FlatScratchBuffer allreduce_scratch_;
};

} // namespace infinilm::models::deepseek_v4
