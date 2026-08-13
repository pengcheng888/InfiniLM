#include "deepseek_v4_mlp.hpp"

#include "../../global_state/global_state.hpp"

#include "infinicore/ops/deepseek_v4_moe_marlin_repack.hpp"
#include "infinicore/ops/deepseek_v4_shared_experts_impl_int8_marlin.hpp"
#include "infinicore/ops/deepseek_v4_silu_and_mul.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::deepseek_v4 {

thread_local DeepseekV4SharedExpertScratch DeepseekV4MLP::shared_scratch_;

namespace {

constexpr size_t kSharedExpertTopK = 1;
constexpr size_t kSharedExpertBlockSize = 16;

size_t shared_expert_padded_tokens(size_t tokens) {
    const size_t flat_topk = tokens * kSharedExpertTopK;
    return ((flat_topk + kSharedExpertBlockSize - 1 + kSharedExpertBlockSize - 1) / kSharedExpertBlockSize) * kSharedExpertBlockSize;
}

} // namespace

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

DeepseekV4PackedMLP::DeepseekV4PackedMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         const infinicore::Device &device)
    : DeepseekV4MLP(std::move(model_config), device) {
}

void DeepseekV4PackedMLP::process_weights_after_loading() {
    DeepseekV4MLP::process_weights_after_loading();

    auto w13_weight = gate_up_proj_->weight();
    auto w2_weight = w2_->weight();
    auto w13_weight_scale = gate_up_proj_->weight_scale();
    auto w2_weight_scale = w2_->weight_scale();
    if (!w13_weight || !w2_weight || !w13_weight_scale || !w2_weight_scale) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4PackedMLP requires int8 weights and weight_scale tensors");
    }
    if (w13_weight->ndim() != 2 || w2_weight->ndim() != 2 || w13_weight_scale->ndim() != 2 || w2_weight_scale->ndim() != 2) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4PackedMLP expects 2D shared MLP weights/scales after loading");
    }
    const size_t gate_up_size = intermediate_size_per_partition_ * 2;
    const size_t hidden_size = w13_weight->size(1);
    if (w13_weight->size(0) != gate_up_size || w2_weight->size(0) != hidden_size || w2_weight->size(1) != intermediate_size_per_partition_) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4PackedMLP unexpected shared MLP weight shape");
    }
    if (hidden_size % 64 != 0 || intermediate_size_per_partition_ % 64 != 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4PackedMLP requires hidden and local intermediate sizes divisible by 64");
    }

    auto w13_weight_3d = w13_weight->view({1, gate_up_size, hidden_size});
    auto w2_weight_3d = w2_weight->view({1, hidden_size, intermediate_size_per_partition_});
    w13_weight_scale_view_ = w13_weight_scale->view({1, gate_up_size, 1});
    w2_weight_scale_view_ = w2_weight_scale->view({1, hidden_size, 1});

    w13_weight_marlin_ = infinicore::Tensor::empty(
        {1, hidden_size / 64, gate_up_size * 64},
        w13_weight->dtype(),
        w13_weight->device());
    w2_weight_marlin_ = infinicore::Tensor::empty(
        {1, intermediate_size_per_partition_ / 64, hidden_size * 64},
        w2_weight->dtype(),
        w2_weight->device());
    infinicore::op::deepseek_v4_moe_marlin_repack_(w13_weight_marlin_, w13_weight_3d);
    infinicore::op::deepseek_v4_moe_marlin_repack_(w2_weight_marlin_, w2_weight_3d);
}

infinicore::Tensor DeepseekV4PackedMLP::forward(infinicore::Tensor hidden_states) const {
    return forward_packed(hidden_states);
}

infinicore::Tensor DeepseekV4PackedMLP::forward_packed(infinicore::Tensor hidden_states) const {
    if (!w13_weight_marlin_ || !w2_weight_marlin_ || !w13_weight_scale_view_ || !w2_weight_scale_view_) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4PackedMLP::forward_packed requires process_weights_after_loading()");
    }
    const size_t tokens = hidden_states->size(0);
    const size_t gate_up_size = intermediate_size_per_partition_ * 2;
    const size_t flat_topk = tokens * kSharedExpertTopK;
    const size_t padded_tokens = shared_expert_padded_tokens(tokens);

    auto output = output_scratch_.get(
        hidden_states->shape(),
        hidden_states->dtype(),
        hidden_states->device());

    auto sorted_token_ids = sorted_token_ids_scratch_.get({padded_tokens}, infinicore::DataType::I32, hidden_states->device());
    auto expert_ids = expert_ids_scratch_.get({padded_tokens / kSharedExpertBlockSize}, infinicore::DataType::I32, hidden_states->device());
    auto num_tokens_post_pad = num_tokens_post_pad_scratch_.get({1}, infinicore::DataType::I32, hidden_states->device());
    auto topk_weights = topk_weights_scratch_.get({tokens, kSharedExpertTopK}, infinicore::DataType::F32, hidden_states->device());

    auto q_hidden = q_hidden_scratch_.get(hidden_states->shape(), infinicore::DataType::I8, hidden_states->device());
    auto hidden_scale = hidden_scale_scratch_.get({tokens, 1}, infinicore::DataType::F32, hidden_states->device());
    auto gate_up = gate_up_scratch_.get({tokens, kSharedExpertTopK, gate_up_size}, hidden_states->dtype(), hidden_states->device());
    auto q_activated = q_activated_scratch_.get({flat_topk, intermediate_size_per_partition_}, infinicore::DataType::I8, hidden_states->device());
    auto activated_scale = activated_scale_scratch_.get({flat_topk, 1}, infinicore::DataType::F32, hidden_states->device());

    infinicore::op::deepseek_v4_shared_experts_impl_int8_marlin_(
        output,
        hidden_states,
        w13_weight_marlin_,
        w2_weight_marlin_,
        w13_weight_scale_view_,
        w2_weight_scale_view_,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        q_hidden,
        hidden_scale,
        gate_up,
        q_activated,
        activated_scale);
    return output;
}

} // namespace infinilm::models::deepseek_v4
