#include "deepseek_v4_for_causal_lm.hpp"

#include "../models_registry.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {

DeepseekV4ForCausalLM::DeepseekV4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                             const infinicore::Device &device) {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto dtype = model_config->get_dtype();

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output DeepseekV4ForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    if (logits->ndim() == 2) {
        logits = logits->view({1, logits->size(0), logits->size(1)});
    }
    return {logits, hidden_states};
}

infinicore::Tensor DeepseekV4ForCausalLM::logits_from_hidden(const infinicore::Tensor &hidden_states) const {
    auto mutable_hidden = hidden_states;
    auto logits = lm_head_->forward(mutable_hidden);
    if (logits->ndim() == 2) {
        logits = logits->view({1, logits->size(0), logits->size(1)});
    }
    return logits;
}

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v4_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("deepseek_v4" != model_type) {
        throw std::runtime_error("infinilm::models::deepseek_v4::create_deepseek_v4_model_config: model_type is not deepseek_v4");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if ((!config_json.contains("quantization_config") || config_json["quantization_config"].is_null()) && config_json.contains("compression_config")) {
        config_json["quantization_config"] = config_json["compression_config"];
    }
    if (config_json.contains("quantization_config") && config_json["quantization_config"].is_object()) {
        config_json["quantization_config"]["quant_method"] = "compressed-tensors";
    }
    if (!config_json.contains("qk_nope_head_dim")) {
        config_json["qk_nope_head_dim"] = config_json.value("head_dim", 512) - config_json.value("qk_rope_head_dim", 64);
    }
    return std::make_shared<infinilm::config::ModelConfig>(config_json);
}

} // namespace infinilm::models::deepseek_v4

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    deepseek_v4,
    infinilm::models::deepseek_v4::DeepseekV4ForCausalLM,
    infinilm::models::deepseek_v4::create_deepseek_v4_model_config);
} // namespace
