#include "glm4_moe_lite_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/deepseek_moe.hpp"
#include "infinicore/ops/deepseek_v4_silu_and_mul.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::glm4_moe_lite {

Glm4MoeLiteSharedMLP::Glm4MoeLiteSharedMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    const auto quantization_method = model_config->get_quantization_method();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size")
                                   * model_config->get_or<size_t>("n_shared_experts", 1);
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    if (tp_size == 0 || intermediate_size % tp_size != 0) {
        throw std::runtime_error("Glm4MoeLiteSharedMLP: intermediate_size must be divisible by tp_size");
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

infinicore::Tensor Glm4MoeLiteSharedMLP::forward(infinicore::Tensor hidden_states) const {
    auto gate_up = gate_up_proj_->forward(hidden_states);
    auto activated = infinicore::Tensor::empty(
        {hidden_states->size(0), intermediate_size_per_partition_},
        gate_up->dtype(),
        gate_up->device());
    infinicore::op::deepseek_v4_silu_and_mul_(activated, gate_up);
    return down_proj_->forward(activated);
}

void Glm4MoeLiteSharedMLP::process_weights_after_loading() {
    gate_up_proj_->process_weights_after_loading();
    down_proj_->process_weights_after_loading();
}

void Glm4MoeLiteSharedMLP::reset_runtime_state() const {
    gate_up_proj_->reset_runtime_state();
    down_proj_->reset_runtime_state();
}

Glm4MoeLiteExperts::Glm4MoeLiteExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    num_experts_ = model_config->get_or_alias<size_t>("n_routed_experts", "num_experts", 0);
    if (num_experts_ == 0) {
        throw std::runtime_error("Glm4MoeLiteExperts: n_routed_experts is required");
    }

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;
    if (tp_size_ == 0 || intermediate_size % tp_size_ != 0) {
        throw std::runtime_error("Glm4MoeLiteExperts: moe_intermediate_size must be divisible by tp_size");
    }
    intermediate_size_per_partition_ = intermediate_size / tp_size_;

    gate_weights_.reserve(num_experts_);
    up_weights_.reserve(num_experts_);
    down_weights_.reserve(num_experts_);
    for (size_t expert = 0; expert < num_experts_; ++expert) {
        const std::string prefix = std::to_string(expert) + ".";
        gate_weights_.push_back(this->register_parameter(
            prefix + "gate_proj.weight",
            infinicore::nn::Parameter({intermediate_size, hidden_size}, dtype, device, 0, tp_rank, tp_size_)));
        up_weights_.push_back(this->register_parameter(
            prefix + "up_proj.weight",
            infinicore::nn::Parameter({intermediate_size, hidden_size}, dtype, device, 0, tp_rank, tp_size_)));
        down_weights_.push_back(this->register_parameter(
            prefix + "down_proj.weight",
            infinicore::nn::Parameter({hidden_size, intermediate_size}, dtype, device, 1, tp_rank, tp_size_)));
    }
}

infinicore::Tensor Glm4MoeLiteExperts::forward(const infinicore::Tensor &hidden_states,
                                               const infinicore::Tensor &topk_indices,
                                               const infinicore::Tensor &topk_weights) const {
    auto output = infinicore::op::deepseek_moe(
        hidden_states,
        topk_indices,
        topk_weights,
        gate_weights_,
        up_weights_,
        down_weights_,
        intermediate_size_per_partition_,
        num_experts_);
    if (tp_size_ > 1 && communicator_ != nullptr) {
        infinicore::op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator_);
    }
    return output;
}

Glm4MoeLiteMoE::Glm4MoeLiteMoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               size_t layer_idx,
                               const infinicore::Device &device) {
    (void)layer_idx;
    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    if (model_config->get_or<size_t>("n_shared_experts", 0) > 0) {
        INFINICORE_NN_MODULE_INIT(shared_experts, model_config, device);
    }
}

infinicore::Tensor Glm4MoeLiteMoE::forward(infinicore::Tensor hidden_states) const {
    auto [routing_weights, selected_experts] = gate_->forward(hidden_states);
    auto routed = experts_->forward(hidden_states, selected_experts, routing_weights);
    if (shared_experts_) {
        auto shared = shared_experts_->forward(hidden_states);
        routed = infinicore::op::add(routed, shared);
    }
    return routed;
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
