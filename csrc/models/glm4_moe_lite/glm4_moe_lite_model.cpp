#include "glm4_moe_lite_model.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::glm4_moe_lite {

Glm4MoeLiteModel::Glm4MoeLiteModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    hidden_size_ = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size_, std::nullopt, dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<Glm4MoeLiteDecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm, hidden_size_, rms_norm_eps, dtype, device);
}

infinicore::Tensor Glm4MoeLiteModel::forward(const infinilm::InfinilmModel::Input &input) const {
    if (!input.input_ids.has_value() || !input.position_ids.has_value()) {
        throw std::runtime_error("Glm4MoeLiteModel::forward requires input_ids and position_ids");
    }
    auto flat_input_ids = input.input_ids.value()->view({input.input_ids.value()->numel()});
    auto positions = input.position_ids.value()->view({input.position_ids.value()->numel()});
    auto hidden_states = embed_tokens_->forward(flat_input_ids);
    if (hidden_states->ndim() != 2) {
        hidden_states = hidden_states->view({flat_input_ids->numel(), hidden_size_});
    }

    for (const auto &layer : layers_) {
        hidden_states = layer->forward(positions, hidden_states);
    }
    return norm_->forward(hidden_states);
}

void Glm4MoeLiteModel::process_weights_after_loading() {
    for (const auto &layer : layers_) {
        layer->process_weights_after_loading();
    }
}

void Glm4MoeLiteModel::reset_runtime_state() const {
    for (const auto &layer : layers_) {
        layer->reset_runtime_state();
    }
}

} // namespace infinilm::models::glm4_moe_lite
