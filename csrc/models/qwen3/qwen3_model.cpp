#include "qwen3_model.hpp"

#include <stdexcept>

namespace infinilm::models::qwen3 {

Qwen3Model::Qwen3Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    size_t vocab_size = model_config->get<size_t>("vocab_size");
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt, dtype, device);

    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<Qwen3DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }

    INFINICORE_NN_MODULE_INIT(norm, hidden_size, rms_norm_eps, dtype, device);
}

infinicore::Tensor Qwen3Model::forward(const infinilm::InfinilmModel::Input &input) const {
    if (!input.input_ids.has_value() || !input.position_ids.has_value()) {
        throw std::runtime_error("infinilm::models::qwen3::Qwen3Model: input_ids and position_ids are required");
    }

    auto hidden_states = embed_tokens_->forward(input.input_ids.value());
    infinicore::Tensor residual;
    for (const auto &layer : layers_) {
        layer->forward(input.position_ids.value(), hidden_states, residual);
    }
    norm_->forward_inplace(hidden_states, residual);
    return hidden_states;
}

infinicore::Tensor Qwen3Model::forward_naive(const infinilm::InfinilmModel::Input &input) const {
    if (!input.input_ids.has_value() || !input.position_ids.has_value()) {
        throw std::runtime_error("infinilm::models::qwen3::Qwen3Model: input_ids and position_ids are required");
    }

    auto hidden_states = embed_tokens_->forward(input.input_ids.value());
    for (const auto &layer : layers_) {
        hidden_states = layer->forward(input.position_ids.value(), hidden_states);
    }
    return norm_->forward(hidden_states);
}

infinicore::Tensor Qwen3Model::embed_tokens(const infinicore::Tensor &input_ids) const {
    return embed_tokens_->forward(input_ids);
}

} // namespace infinilm::models::qwen3
