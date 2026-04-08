#pragma once

#include "../../config/model_config.hpp"
#include <memory>

namespace infinilm::layers::rotary_embedding {

std::shared_ptr<infinicore::nn::RoPE> get_rope(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                                               const infinicore::Device &device);

} // namespace infinilm::layers::rotary_embedding
