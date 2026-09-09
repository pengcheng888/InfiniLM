#pragma once

#include "../../layers/linear/linear.hpp"
#include "../../models/infinilm_model.hpp"
#include "deepseek_v4_decoder_layer.hpp"
#include "deepseek_v4_model.hpp"

#include <memory>
#include <utility>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4ForCausalLM : public infinilm::InfinilmModel {
public:
    DeepseekV4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          const infinicore::Device &device);

    void reset_cache(const cache::CacheConfig *cache_config) override;
    Output forward(const Input &input) const override;
    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(DeepseekV4Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);

    infinicore::Device device_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::deepseek_v4
