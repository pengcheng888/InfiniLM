#include "glm4_moe_lite_decoder_layer.hpp"

namespace infinilm::models::glm4_moe_lite {

Glm4MoeLiteDecoderLayer::Glm4MoeLiteDecoderLayer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    if (layer_idx < model_config->get_or<size_t>("first_k_dense_replace", 1)) {
        INFINICORE_NN_MODULE_INIT(mlp, model_config, device);
    } else {
        moe_ = this->register_module<Glm4MoeLiteMoE>("mlp", model_config, layer_idx, device);
    }
    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Glm4MoeLiteDecoderLayer::forward(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual) const {
    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    if (mlp_) {
        const auto hidden_shape = hidden_states->shape();
        if (hidden_shape.size() == 3) {
            hidden_states = mlp_->forward(hidden_states);
        } else {
            auto mlp_input = hidden_states->view({1, hidden_shape[0], hidden_shape[1]});
            hidden_states = mlp_->forward(mlp_input)->view(hidden_shape);
        }
    } else {
        hidden_states = moe_->forward(hidden_states);
    }
    return {hidden_states, residual};
}

void Glm4MoeLiteDecoderLayer::process_weights_after_loading() {
    self_attn_->process_weights_after_loading();
    if (mlp_) {
        mlp_->process_weights_after_loading();
    }
    if (moe_) {
        moe_->process_weights_after_loading();
    }
}

void Glm4MoeLiteDecoderLayer::reset_runtime_state() const {
    self_attn_->reset_runtime_state();
    if (mlp_) {
        mlp_->reset_runtime_state();
    }
    if (moe_) {
        moe_->reset_runtime_state();
    }
}

} // namespace infinilm::models::glm4_moe_lite
