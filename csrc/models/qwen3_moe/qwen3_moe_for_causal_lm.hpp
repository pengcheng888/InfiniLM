#pragma once

#include "qwen3_moe_sparse_moe_block.hpp"
#include <memory>
#include "../../layers/common_modules.hpp"

namespace infinilm::models::qwen3_moe {

using Qwen3MoeAttention = infinilm::layers::attention::Attention;

using Qwen3MoeDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<Qwen3MoeAttention, Qwen3MoeSparseMoeBlock>;

using Qwen3MoeModel = infinilm::layers::causal_lm_templates::TextModel<Qwen3MoeDecoderLayer>;

using Qwen3MoeForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<Qwen3MoeModel>;

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_moe
