#include "glm4_moe_lite_mlp.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops/deepseek_v4_silu_and_mul.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::glm4_moe_lite {

Glm4MoeLiteMLP::Glm4MoeLiteMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    const auto quantization_method = model_config->get_quantization_method();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("intermediate_size");
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    if (tp_size == 0 || intermediate_size % tp_size != 0) {
        throw std::runtime_error("Glm4MoeLiteMLP: intermediate_size must be divisible by tp_size");
    }
    intermediate_size_per_partition_ = intermediate_size / tp_size;

    auto register_fn = [this](const std::string &name, infinicore::nn::Parameter param) {
        this->register_parameter(name, std::move(param));
    };
    gate_up_proj_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
        hidden_size,
        intermediate_size,
        "gate_proj",
        "up_proj",
        register_fn,
        quantization_method,
        false,
        dtype,
        device,
        rank_info);
    down_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "down_proj",
        intermediate_size,
        hidden_size,
        quantization_method,
        false,
        dtype,
        device,
        tp_rank,
        tp_size,
        rank_info.comm);
}

infinicore::Tensor Glm4MoeLiteMLP::forward(infinicore::Tensor hidden_states) const {
    auto gate_up = gate_up_proj_->forward(hidden_states);
    auto activated = infinicore::Tensor::empty(
        {hidden_states->size(0), intermediate_size_per_partition_},
        gate_up->dtype(),
        gate_up->device());
    infinicore::op::deepseek_v4_silu_and_mul_(activated, gate_up);
    return down_proj_->forward(activated);
}

void Glm4MoeLiteMLP::process_weights_after_loading() {
    gate_up_proj_->process_weights_after_loading();
    down_proj_->process_weights_after_loading();
}

void Glm4MoeLiteMLP::reset_runtime_state() const {
    gate_up_proj_->reset_runtime_state();
    down_proj_->reset_runtime_state();
}

} // namespace infinilm::models::glm4_moe_lite
