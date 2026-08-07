#pragma once

#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"
#include "qwen3_model.hpp"
#include <memory>

namespace infinilm::models::qwen3 {

class Qwen3ForCausalLM : public InfinilmModel {
public:
    Qwen3ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const infinicore::Device &device);

    Output forward(const Input &input) const override;
    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(Qwen3Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3
