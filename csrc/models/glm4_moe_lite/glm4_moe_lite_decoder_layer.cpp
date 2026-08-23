#include "glm4_moe_lite_decoder_layer.hpp"

#include "infinicore/ops.hpp"

#include <stdexcept>

namespace infinilm::models::glm4_moe_lite {

Glm4MoeLiteDecoderLayer::Glm4MoeLiteDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 size_t layer_idx,
                                                 const infinicore::Device &device)
    : layer_idx_(layer_idx),
      first_k_dense_replace_(model_config->get_or<size_t>("first_k_dense_replace", 1)) {
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    if (layer_idx_ < first_k_dense_replace_) {
        INFINICORE_NN_MODULE_INIT(mlp, model_config, device);
    } else {
        moe_ = this->register_module<Glm4MoeLiteMoE>("mlp", model_config, layer_idx, device);
    }
    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
}

infinicore::Tensor Glm4MoeLiteDecoderLayer::forward(const infinicore::Tensor &positions,
                                                    infinicore::Tensor hidden_states) const {
    auto residual = hidden_states;
    auto attn_in = input_layernorm_->forward(hidden_states);
    auto attn_out = self_attn_->forward(positions, attn_in);
    hidden_states = infinicore::op::add(residual, attn_out);

    residual = hidden_states;
    auto mlp_in = post_attention_layernorm_->forward(hidden_states);
    auto mlp_out = mlp_ ? mlp_->forward(mlp_in) : moe_->forward(mlp_in);
    return infinicore::op::add(residual, mlp_out);
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
