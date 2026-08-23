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

    virtual infinicore::Tensor forward(infinicore::Tensor hidden_states) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

protected:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> w2_;
    static thread_local DeepseekV4SharedExpertScratch shared_scratch_;
    size_t intermediate_size_per_partition_{0};
    infinicore::DataType dtype_{infinicore::DataType::BF16};
    infinicore::Device device_;
};

class DeepseekV4PackedMLP : public DeepseekV4MLP {
public:
    DeepseekV4PackedMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        const infinicore::Device &device);

    infinicore::Tensor forward(infinicore::Tensor hidden_states) const override;
    infinicore::Tensor forward_packed(infinicore::Tensor hidden_states) const;

    void process_weights_after_loading() override;

private:
    mutable DeepseekV4FlatScratchBuffer output_scratch_;
    mutable DeepseekV4FlatScratchBuffer sorted_token_ids_scratch_;
    mutable DeepseekV4FlatScratchBuffer expert_ids_scratch_;
    mutable DeepseekV4FlatScratchBuffer num_tokens_post_pad_scratch_;
    mutable DeepseekV4FlatScratchBuffer topk_weights_scratch_;
    mutable DeepseekV4FlatScratchBuffer q_hidden_scratch_;
    mutable DeepseekV4FlatScratchBuffer hidden_scale_scratch_;
    mutable DeepseekV4FlatScratchBuffer gate_up_scratch_;
    mutable DeepseekV4FlatScratchBuffer q_activated_scratch_;
    mutable DeepseekV4FlatScratchBuffer activated_scale_scratch_;
    infinicore::Tensor w13_weight_marlin_;
    infinicore::Tensor w2_weight_marlin_;
    infinicore::Tensor w13_weight_scale_view_;
    infinicore::Tensor w2_weight_scale_view_;
};

} // namespace infinilm::models::deepseek_v4
