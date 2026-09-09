#include "deepseek_v4_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops/deepseek_v4/moe_w8a8.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"

#include <infiniccl.h>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {
namespace {

std::shared_ptr<infinilm::config::ModelConfig> make_shared_experts_config(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    auto config_json = model_config->get_config_json();
    config_json["intermediate_size"] = model_config->get<size_t>("moe_intermediate_size")
                                     * model_config->get_or<size_t>("n_shared_experts", 1);
    return std::make_shared<infinilm::config::ModelConfig>(config_json);
}

} // namespace

DeepseekV4PackedExperts::DeepseekV4PackedExperts(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    tp_size_ = rank_info.tp_size;

    num_experts_ = model_config->get_or_alias<size_t>("n_routed_experts", "num_experts", 0);
    hidden_size_ = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    if (num_experts_ == 0) {
        throw std::runtime_error("DeepseekV4PackedExperts: n_routed_experts/num_experts is required");
    }
    if (tp_size_ <= 0 || intermediate_size % static_cast<size_t>(tp_size_) != 0) {
        throw std::runtime_error("DeepseekV4PackedExperts: moe_intermediate_size must be divisible by tp_size");
    }
    intermediate_size_per_partition_ = intermediate_size / static_cast<size_t>(tp_size_);

    INFINICORE_NN_PARAMETER_INIT(
        w13_weight,
        ({num_experts_, 2 * intermediate_size_per_partition_, hidden_size_}, infinicore::DataType::I8, device));
    INFINICORE_NN_PARAMETER_INIT(
        w13_weight_scale,
        ({num_experts_, 2 * intermediate_size_per_partition_, 1}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(
        w2_weight,
        ({num_experts_, hidden_size_, intermediate_size_per_partition_}, infinicore::DataType::I8, device));
    INFINICORE_NN_PARAMETER_INIT(
        w2_weight_scale,
        ({num_experts_, hidden_size_, 1}, infinicore::DataType::F32, device));

    for (size_t expert = 0; expert < num_experts_; ++expert) {
        const std::string prefix = std::to_string(expert) + ".";
        auto w1 = w13_weight_
                      ->narrow({{0, expert, 1}, {1, 0, intermediate_size_per_partition_}})
                      ->squeeze(0);
        auto w3 = w13_weight_
                      ->narrow({{0, expert, 1}, {1, intermediate_size_per_partition_, intermediate_size_per_partition_}})
                      ->squeeze(0);
        auto w1_scale = w13_weight_scale_
                            ->narrow({{0, expert, 1}, {1, 0, intermediate_size_per_partition_}})
                            ->squeeze(0);
        auto w3_scale = w13_weight_scale_
                            ->narrow({{0, expert, 1}, {1, intermediate_size_per_partition_, intermediate_size_per_partition_}})
                            ->squeeze(0);
        auto w2 = w2_weight_
                      ->narrow({{0, expert, 1}})
                      ->squeeze(0);
        auto w2_scale = w2_weight_scale_
                            ->narrow({{0, expert, 1}})
                            ->squeeze(0);

        this->register_parameter(prefix + "w1.weight", infinicore::nn::Parameter(w1, 0, tp_rank, tp_size_));
        this->register_parameter(prefix + "w3.weight", infinicore::nn::Parameter(w3, 0, tp_rank, tp_size_));
        this->register_parameter(prefix + "w1.weight_scale", infinicore::nn::Parameter(w1_scale, 0, tp_rank, tp_size_));
        this->register_parameter(prefix + "w3.weight_scale", infinicore::nn::Parameter(w3_scale, 0, tp_rank, tp_size_));
        this->register_parameter(prefix + "w2.weight", infinicore::nn::Parameter(w2, 1, tp_rank, tp_size_));
        this->register_parameter(prefix + "w2.weight_scale", infinicore::nn::Parameter(w2_scale));
    }
}

infinicore::Tensor DeepseekV4PackedExperts::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &topk_indices,
    const infinicore::Tensor &topk_weights) const {
    ASSERT(hidden_states->ndim() == 2);
    ASSERT(topk_indices->ndim() == 2 && topk_weights->ndim() == 2);

    auto output = infinicore::Tensor::empty(
        {hidden_states->size(0), hidden_size_},
        hidden_states->dtype(),
        hidden_states->device());
    infinicore::op::deepseek_v4::moe_w8a8_(
        output,
        hidden_states,
        topk_weights,
        topk_indices,
        w13_weight_,
        w13_weight_scale_,
        w2_weight_,
        w2_weight_scale_,
        10.0);
    return output;
}

DeepseekV4MoE::DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             size_t layer_idx,
                             const infinicore::Device &device) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = rank_info.tp_size;
    communicator_ = rank_info.comm;
    INFINICORE_NN_MODULE_INIT(gate, model_config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    if (model_config->get_or<size_t>("n_shared_experts", 0) > 0) {
        shared_experts_ = this->register_module<DeepseekV4MLP>(
            "shared_experts", make_shared_experts_config(model_config), device);
    }
}

infinicore::Tensor DeepseekV4MoE::forward(infinicore::Tensor hidden_states,
                                          const infinicore::Tensor &input_ids) const {
    const auto shape = hidden_states->shape();
    const bool restore_3d_shape = shape.size() == 3;
    auto flat_hidden_states = restore_3d_shape
                                ? hidden_states->view({shape[0] * shape[1], shape[2]})
                                : hidden_states;

    auto [routing_weights, selected_experts] = gate_->forward(flat_hidden_states, input_ids);
    auto routed = experts_->forward(flat_hidden_states, selected_experts, routing_weights);
    if (shared_experts_) {
        auto shared = shared_experts_->forward(flat_hidden_states);
        routed = infinicore::op::add(routed, shared);
    }
    if (tp_size_ > 1 && communicator_ != nullptr) {
        infinicore::op::distributed::allreduce_(routed, routed, INFINICCL_SUM, communicator_);
    }
    return restore_3d_shape ? routed->view(shape) : routed;
}

void DeepseekV4MoE::process_weights_after_loading() {
    if (shared_experts_) {
        shared_experts_->process_weights_after_loading();
    }
}

void DeepseekV4MoE::reset_runtime_state() const {
    if (shared_experts_) {
        shared_experts_->reset_runtime_state();
    }
}

} // namespace infinilm::models::deepseek_v4
