#include "glm4_moe_lite_moe.hpp"

#include "infinicore/ops/add.hpp"
#include "infinicore/ops/mul_scalar.hpp"

namespace infinilm::models::glm4_moe_lite {
namespace {

std::shared_ptr<infinilm::config::ModelConfig> make_shared_experts_config(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    auto config_json = model_config->get_config_json();
    config_json["intermediate_size"] = model_config->get<size_t>("moe_intermediate_size")
                                     * model_config->get_or<size_t>("n_shared_experts", 1);
    return std::make_shared<infinilm::config::ModelConfig>(config_json);
}

} // namespace

Glm4MoeLiteMoE::Glm4MoeLiteMoE(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : routed_scaling_factor_(model_config->get_or<float>("routed_scaling_factor", 1.0f)) {
    (void)layer_idx;
    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    if (model_config->get_or<size_t>("n_shared_experts", 0) > 0) {
        shared_experts_ = this->register_module<Glm4MoeLiteMLP>(
            "shared_experts", make_shared_experts_config(model_config), device);
    }
}

infinicore::Tensor Glm4MoeLiteMoE::forward(infinicore::Tensor hidden_states) const {
    const auto hidden_shape = hidden_states->shape();
    const bool restore_3d_shape = hidden_shape.size() == 3;
    auto flat_hidden_states = restore_3d_shape
                                ? hidden_states->view({hidden_shape[0] * hidden_shape[1], hidden_shape[2]})
                                : hidden_states;

    auto [routing_weights, selected_experts] = gate_->forward(flat_hidden_states);
    auto routed = experts_->forward(flat_hidden_states, selected_experts, routing_weights);
    if (routed_scaling_factor_ != 1.0f) {
        routed = infinicore::op::mul_scalar(routed, routed_scaling_factor_);
    }
    if (shared_experts_) {
        const auto flat_shape = flat_hidden_states->shape();
        auto shared_input = flat_hidden_states->view({1, flat_shape[0], flat_shape[1]});
        auto shared = shared_experts_->forward(shared_input)->view(flat_shape);
        routed = infinicore::op::add(routed, shared);
    }
    return restore_3d_shape ? routed->view(hidden_shape) : routed;
}

void Glm4MoeLiteMoE::process_weights_after_loading() {
    if (shared_experts_) {
        shared_experts_->process_weights_after_loading();
    }
}

void Glm4MoeLiteMoE::reset_runtime_state() const {
    if (shared_experts_) {
        shared_experts_->reset_runtime_state();
    }
}

} // namespace infinilm::models::glm4_moe_lite
