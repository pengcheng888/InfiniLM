#pragma once

#include "../../layers/lm_head/parallel_lm_head.hpp"
#include "../../models/infinilm_model.hpp"
#include "glm4_moe_lite_model.hpp"

#include <memory>

namespace infinilm::models::glm4_moe_lite {

class Glm4MoeLiteForCausalLM : public infinilm::InfinilmModel {
public:
    Glm4MoeLiteForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device);

    infinilm::InfinilmModel::Output forward(const infinilm::InfinilmModel::Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;
    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const;

private:
    infinicore::Tensor compute_lm_head_logits(const infinicore::Tensor &hidden_states) const;

    INFINICORE_NN_MODULE(Glm4MoeLiteModel, model);
    std::shared_ptr<infinilm::layers::lm_head::ParallelLMHead> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_glm4_moe_lite_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::glm4_moe_lite
