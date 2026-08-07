#include "deepseek_v4_mlp.hpp"

#include "../../global_state/global_state.hpp"

#include "infinicore/ops/deepseek_v4_silu_and_mul.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::deepseek_v4 {

thread_local DeepseekV4SharedExpertScratch DeepseekV4MLP::shared_scratch_;

DeepseekV4MLP::DeepseekV4MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    dtype_ = dtype;
    device_ = device;
    const auto quantization_method = model_config->get_quantization_method();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size") * model_config->get_or<size_t>("n_shared_experts", 1);
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    if (tp_size == 0 || intermediate_size % tp_size != 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4MLP: intermediate_size must be divisible by tp_size");
    }
    intermediate_size_per_partition_ = intermediate_size / tp_size;

    auto register_fn = [this](const std::string &name, infinicore::nn::Parameter param) {
        this->register_parameter(name, std::move(param));
    };
    gate_up_proj_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
        hidden_size,
        intermediate_size,
        "w1",
        "w3",
        register_fn,
        quantization_method,
        false,
        dtype,
        device,
        rank_info);
    w2_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "w2", intermediate_size, hidden_size, quantization_method, false, dtype, device, tp_rank, tp_size, nullptr);
}

infinicore::Tensor DeepseekV4MLP::forward(infinicore::Tensor hidden_states) const {
    auto gate_up = shared_scratch_.get_gate_up(
        {hidden_states->size(0), intermediate_size_per_partition_ * 2},
        hidden_states->dtype(),
        hidden_states->device());
    gate_up_proj_->forward_(gate_up, hidden_states);
    auto activated = shared_scratch_.get_activated(
        {hidden_states->size(0), intermediate_size_per_partition_},
        gate_up->dtype(),
        gate_up->device());
    infinicore::op::deepseek_v4_silu_and_mul_(activated, gate_up);
    return w2_->forward(activated);
}

void DeepseekV4MLP::process_weights_after_loading() {
    gate_up_proj_->process_weights_after_loading();
    w2_->process_weights_after_loading();
    shared_scratch_.preallocate_scratch(intermediate_size_per_partition_, dtype_, device_);
}

void DeepseekV4MLP::reset_runtime_state() const {
    gate_up_proj_->reset_runtime_state();
    w2_->reset_runtime_state();
}

} // namespace infinilm::models::deepseek_v4
