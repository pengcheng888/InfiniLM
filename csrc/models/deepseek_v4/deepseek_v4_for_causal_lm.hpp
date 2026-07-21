#pragma once

#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"
#include "deepseek_v4_model.hpp"

#include <memory>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4ForCausalLM : public InfinilmModel {
public:
    DeepseekV4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          const infinicore::Device &device);

    Output forward(const Input &input) const override;
    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(DeepseekV4Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v4_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::deepseek_v4
