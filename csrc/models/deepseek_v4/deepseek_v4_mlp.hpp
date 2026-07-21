#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <tuple>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4MoEGate : public infinicore::nn::Module {
public:
    DeepseekV4MoEGate(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t layer_idx,
                      const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &hidden_states,
                                                               const infinicore::Tensor &input_ids) const;

private:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(tid2eid);
    INFINICORE_NN_PARAMETER(bias);

    size_t num_experts_per_tok_{0};
    bool norm_topk_prob_{true};
    bool is_hash_{true};
};

class DeepseekV4SharedExperts : public infinicore::nn::Module {
public:
    DeepseekV4SharedExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            const infinicore::Device &device);

    infinicore::Tensor forward(infinicore::Tensor hidden_states) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> w1_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> w2_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> w3_;
};

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
                               const infinicore::Tensor &topk_indices) const;

    void process_weights_after_loading() override;

private:
    infinicore::Tensor forward_reference(const infinicore::Tensor &hidden_states,
                                         const infinicore::Tensor &topk_weights,
                                         const infinicore::Tensor &topk_indices) const;
    infinicore::Tensor forward_marlin(const infinicore::Tensor &hidden_states,
                                      const infinicore::Tensor &topk_weights,
                                      const infinicore::Tensor &topk_indices) const;
    bool use_marlin_backend() const;

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
    bool use_marlin_backend_{false};
    int marlin_block_size_{16};
    int marlin_mode_{54};
    int marlin_delta_{1};
    mutable infinicore::Tensor w13_weight_marlin_;
    mutable infinicore::Tensor w2_weight_marlin_;
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
    INFINICORE_NN_MODULE(DeepseekV4SharedExperts, shared_experts);

    size_t layer_idx_{0};
    size_t tp_size_{1};
    infinicclComm_t communicator_{nullptr};
};

} // namespace infinilm::models::deepseek_v4
