#include "qwen3_for_causal_lm.hpp"

#include "../models_registry.hpp"
#include <stdexcept>
#include <string>

namespace infinilm::models::qwen3 {

Qwen3ForCausalLM::Qwen3ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   const infinicore::Device &device) {
    model_config_ = model_config;
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype = model_config->get_dtype();

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output Qwen3ForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

infinicore::Tensor Qwen3ForCausalLM::logits_from_hidden(const infinicore::Tensor &hidden_states) const {
    auto mutable_hidden = hidden_states;
    return lm_head_->forward(mutable_hidden);
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3::create_qwen3_model_config: model_type is not qwen3");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (!config_json.contains("head_dim")) {
        config_json["head_dim"] = model_config->get<size_t>("hidden_size") / model_config->get<size_t>("num_attention_heads");
    }
    if (!config_json.contains("num_key_value_heads")) {
        config_json["num_key_value_heads"] = model_config->get<size_t>("num_attention_heads");
    }
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    if (!config_json.contains("attention_output_bias")) {
        config_json["attention_output_bias"] = false;
    }
    if (!config_json.contains("mlp_bias")) {
        config_json["mlp_bias"] = false;
    }
    if (!config_json.contains("rope_theta")) {
        config_json["rope_theta"] = 1000000.0;
    }
    if (!config_json.contains("max_position_embeddings")) {
        config_json["max_position_embeddings"] = 40960;
    }
    return model_config;
}

} // namespace infinilm::models::qwen3

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3,
    infinilm::models::qwen3::Qwen3ForCausalLM,
    infinilm::models::qwen3::create_qwen3_model_config);
} // namespace
