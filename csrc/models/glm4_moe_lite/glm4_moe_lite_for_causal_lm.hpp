#pragma once

#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "../../layers/causal_lm_templates/text_model.hpp"
#include "glm4_moe_lite_decoder_layer.hpp"

#include <memory>
#include <utility>

namespace infinilm::models::glm4_moe_lite {

using Glm4MoeLiteModel = infinilm::layers::causal_lm_templates::TextModel<Glm4MoeLiteDecoderLayer>;

class Glm4MoeLiteForCausalLM : public infinilm::layers::causal_lm_templates::TextCausalLM<Glm4MoeLiteModel> {
public:
    using Base = infinilm::layers::causal_lm_templates::TextCausalLM<Glm4MoeLiteModel>;

    Glm4MoeLiteForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device)
        : Base(std::move(model_config), device), device_(device) {}

    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    infinicore::Device device_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_glm4_moe_lite_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::glm4_moe_lite
