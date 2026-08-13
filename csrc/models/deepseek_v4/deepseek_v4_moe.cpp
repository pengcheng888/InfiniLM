#include "deepseek_v4_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/moe/ep/ep_config.hpp"
#include "deepseek_v4_profile.hpp"
#include "deepseek_v4_utils.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/deepseek_v4_moe_marlin_repack.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"

#include <infiniccl.h>
#include <stdexcept>
#include <string>

namespace infinicore::op {
bool deepseek_v4_dcu_custom_allreduce_(infinicore::Tensor output,
                                       const infinicore::Tensor &input,
                                       int tp_rank,
                                       int tp_size,
                                       int max_size_bytes = 8192 * 512);
}

namespace infinilm::models::deepseek_v4 {

thread_local DeepseekV4RoutedExpertScratch DeepseekV4PackedExperts::shared_scratch_;

namespace {

void debug_dump_tensor(const infinicore::Tensor &tensor, size_t layer_idx, const std::string &name, bool enabled) {
    if (!enabled || !tensor) {
        return;
    }
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tensor->debug("/tmp/infinilm_dsv4_tp" + std::to_string(rank_info.tp_rank) + "_l" + std::to_string(layer_idx) + "_" + name + ".bin");
}

} // namespace

DeepseekV4PackedExperts::DeepseekV4PackedExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 const infinicore::Device &device) {
    dtype_ = model_config->get_dtype();
    device_ = device;
    num_experts_ = model_config->get_or_alias<size_t>("n_routed_experts", "num_experts", 0);
    hidden_size_ = model_config->get<size_t>("hidden_size");
    intermediate_size_ = model_config->get<size_t>("moe_intermediate_size");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    routed_scaling_factor_ = model_config->get_or<double>("routed_scaling_factor", 1.0);
    if (num_experts_ == 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4PackedExperts: n_routed_experts is required");
    }

    const auto ep_config = infinilm::layers::moe::make_ep_config();
    const auto expert_placement = infinilm::layers::moe::make_expert_placement(ep_config, num_experts_);
    const size_t num_local_experts = expert_placement.local_num_experts;
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    const bool ep_enabled = ep_config.backend != infinilm::layers::moe::EPBackend::Disabled;
    routed_expert_backend_ = moe_backends::select_routed_expert_backend();
    marlin_gemm_override_ = moe_backends::read_marlin_gemm_override_from_env();
    if (routed_expert_backend_.backend == moe_backends::RoutedExpertBackend::FusedExpertsInt8Marlin && tp_size != 8) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4PackedExperts: routed expert backend fused_experts_int8_marlin requires tp_size == 8");
    }

    if (ep_enabled) {
        intermediate_size_per_partition_ = intermediate_size_;
    } else {
        if (tp_size == 0 || intermediate_size_ % tp_size != 0) {
            throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4PackedExperts: intermediate_size must be divisible by tp_size");
        }
        intermediate_size_per_partition_ = intermediate_size_ / tp_size;
    }
    const size_t expert_tp_rank = ep_enabled ? 0 : tp_rank;
    const size_t expert_tp_size = ep_enabled ? 1 : tp_size;

    w13_weight_ = infinicore::nn::Parameter(
        {num_local_experts, intermediate_size_per_partition_ * 2, hidden_size_},
        infinicore::DataType::I8,
        device);
    w13_weight_scale_ = infinicore::nn::Parameter(
        {num_local_experts, intermediate_size_per_partition_ * 2, 1},
        infinicore::DataType::F32,
        device);
    w2_weight_ = infinicore::nn::Parameter(
        {num_local_experts, hidden_size_, intermediate_size_per_partition_},
        infinicore::DataType::I8,
        device);
    w2_weight_scale_ = infinicore::nn::Parameter(
        {num_local_experts, hidden_size_, 1},
        infinicore::DataType::F32,
        device);

    this->register_parameter("w13_weight", w13_weight_);
    this->register_parameter("w13_weight_scale", w13_weight_scale_);
    this->register_parameter("w2_weight", w2_weight_);
    this->register_parameter("w2_weight_scale", w2_weight_scale_);

    for (size_t local_expert = 0; local_expert < num_local_experts; ++local_expert) {
        const size_t global_expert = expert_placement.local_expert_start + local_expert;
        const std::string prefix = std::to_string(global_expert) + ".";

        auto w1 = w13_weight_
                      ->narrow({{0, local_expert, 1}, {1, 0, intermediate_size_per_partition_}})
                      ->squeeze(0);
        auto w3 = w13_weight_
                      ->narrow({{0, local_expert, 1}, {1, intermediate_size_per_partition_, intermediate_size_per_partition_}})
                      ->squeeze(0);
        auto w1_scale = w13_weight_scale_
                            ->narrow({{0, local_expert, 1}, {1, 0, intermediate_size_per_partition_}})
                            ->squeeze(0);
        auto w3_scale = w13_weight_scale_
                            ->narrow({{0, local_expert, 1}, {1, intermediate_size_per_partition_, intermediate_size_per_partition_}})
                            ->squeeze(0);
        auto w2 = w2_weight_
                      ->narrow({{0, local_expert, 1}})
                      ->squeeze(0);
        auto w2_scale = w2_weight_scale_
                            ->narrow({{0, local_expert, 1}})
                            ->squeeze(0);

        this->register_parameter(prefix + "w1.weight", infinicore::nn::Parameter(w1, 0, expert_tp_rank, expert_tp_size));
        this->register_parameter(prefix + "w3.weight", infinicore::nn::Parameter(w3, 0, expert_tp_rank, expert_tp_size));
        this->register_parameter(prefix + "w1.weight_scale", infinicore::nn::Parameter(w1_scale, 0, expert_tp_rank, expert_tp_size));
        this->register_parameter(prefix + "w3.weight_scale", infinicore::nn::Parameter(w3_scale, 0, expert_tp_rank, expert_tp_size));
        this->register_parameter(prefix + "w2.weight", infinicore::nn::Parameter(w2, 1, expert_tp_rank, expert_tp_size));
        this->register_parameter(prefix + "w2.weight_scale", infinicore::nn::Parameter(w2_scale));
    }
}

void DeepseekV4PackedExperts::process_weights_after_loading() {
    shared_scratch_.preallocate_scratch(hidden_size_, dtype_, device_);
    if (!moe_backends::requires_marlin_repack(routed_expert_backend_.backend) || marlin_only_weights_) {
        return;
    }
    const auto process_config = moe_backends::select_marlin_gemm_config(
        1,
        hidden_size_,
        intermediate_size_per_partition_,
        num_experts_per_tok_,
        marlin_block_size_,
        marlin_mode_,
        marlin_delta_,
        marlin_gemm_override_);
    if (!process_config.supported) {
        if (routed_expert_backend_.explicit_backend) {
            throw std::runtime_error(std::string("infinilm::models::deepseek_v4::DeepseekV4PackedExperts: routed expert backend ") + moe_backends::to_string(routed_expert_backend_.backend) + " does not support Marlin repack for this shape");
        }
        return;
    }
    if (hidden_size_ % 64 != 0 || intermediate_size_per_partition_ % 64 != 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4PackedExperts: Marlin W8A8 requires hidden and local intermediate sizes divisible by 64");
    }
    const size_t local_experts = w13_weight_->size(0);
    const size_t gate_up_size = intermediate_size_per_partition_ * 2;
    w13_weight_marlin_ = infinicore::Tensor::empty(
        {local_experts, hidden_size_ / 64, gate_up_size * 64},
        w13_weight_->dtype(),
        w13_weight_->device());
    w2_weight_marlin_ = infinicore::Tensor::empty(
        {local_experts, intermediate_size_per_partition_ / 64, hidden_size_ * 64},
        w2_weight_->dtype(),
        w2_weight_->device());
    infinicore::op::deepseek_v4_moe_marlin_repack_(w13_weight_marlin_, w13_weight_);
    infinicore::op::deepseek_v4_moe_marlin_repack_(w2_weight_marlin_, w2_weight_);

    parameters_.clear();
    infinicore::context::syncStream();
    w13_weight_.reset();
    w2_weight_.reset();
    marlin_only_weights_ = true;
}

moe_backends::RoutedExpertContext DeepseekV4PackedExperts::make_backend_context() const {
    return {
        num_experts_,
        hidden_size_,
        intermediate_size_per_partition_,
        num_experts_per_tok_,
        routed_scaling_factor_,
        marlin_block_size_,
        marlin_mode_,
        marlin_delta_,
        marlin_gemm_override_,
        w13_weight_,
        w13_weight_scale_,
        w2_weight_,
        w2_weight_scale_,
        w13_weight_marlin_,
        w2_weight_marlin_,
        marlin_only_weights_,
    };
}

infinicore::Tensor DeepseekV4PackedExperts::forward(const infinicore::Tensor &hidden_states,
                                                    const infinicore::Tensor &topk_weights,
                                                    const infinicore::Tensor &topk_indices,
                                                    const std::optional<infinicore::Tensor> &shared_output) const {
    return moe_backends::forward_routed_experts(
        routed_expert_backend_,
        make_backend_context(),
        shared_scratch_,
        hidden_states,
        topk_weights,
        topk_indices,
        shared_output);
}

DeepseekV4MoE::DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             size_t layer_idx,
                             const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    debug_dump_enabled_ = utils::debug_dump_enabled();
    moe_allreduce_outplace_enabled_ = utils::moe_allreduce_outplace_enabled();
    moe_custom_allreduce_enabled_ = utils::moe_custom_allreduce_enabled();
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_rank_ = rank_info.tp_rank;
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;
    INFINICORE_NN_MODULE_INIT(gate, model_config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    if (model_config->get_or<size_t>("n_shared_experts", 0) > 0) {
        INFINICORE_NN_MODULE_INIT(shared_experts, model_config, device);
    }
}

infinicore::Tensor DeepseekV4MoE::forward(const infinicore::Tensor &hidden_states,
                                          const infinicore::Tensor &input_ids) const {
    const size_t token_count = hidden_states->size(0);
    profile::ScopedTimer moe_timer(profile::Event::MoeForward, token_count);
    debug_dump_tensor(hidden_states, layer_idx_, "moe_input", debug_dump_enabled_);
    infinicore::Tensor routing_weights;
    infinicore::Tensor selected_experts;
    std::tie(routing_weights, selected_experts) = gate_->forward(hidden_states, input_ids);
    debug_dump_tensor(routing_weights, layer_idx_, "topk_weights", debug_dump_enabled_);
    debug_dump_tensor(selected_experts, layer_idx_, "topk_indices", debug_dump_enabled_);

    infinicore::Tensor shared;
    if (shared_experts_) {
        profile::ScopedTimer timer(profile::Event::MoeSharedExperts, token_count);

        const int repeats = 1;
        for (int i = 0; i < repeats; ++i) {
            bool use_packed_shared_mlp = true;
            if (use_packed_shared_mlp) {
                // 1000   total_ms=135.761 ==> 简单优化后108.246ms => topk从6变为1后，再变为99 ms
                shared = shared_experts_->forward_packed(hidden_states);
            } else {
                // 1000   total_ms=160.761
                shared = shared_experts_->forward_linear(hidden_states);
            }
        }

        debug_dump_tensor(shared, layer_idx_, "shared", debug_dump_enabled_);
    }

    infinicore::Tensor routed;
    {
        profile::ScopedTimer timer(profile::Event::MoeExperts, token_count);
        routed = experts_->forward(
            hidden_states,    // [ntoken, hidden_size]
            routing_weights,  // [ntoken, 6]
            selected_experts, // [ntoken, 6]
            shared ? std::optional<infinicore::Tensor>(shared) : std::nullopt);
    }
    debug_dump_tensor(routed, layer_idx_, shared ? "after_shared" : "routed", debug_dump_enabled_);
    if (tp_size_ > 1 && communicator_ != nullptr) {
        profile::ScopedTimer timer(profile::Event::MoeAllReduce, token_count);
        bool reduced_by_custom = false;
        if (moe_custom_allreduce_enabled_) {
            auto reduced = allreduce_scratch_.get(
                routed->shape(),
                routed->dtype(),
                routed->device());
            reduced_by_custom = infinicore::op::deepseek_v4_dcu_custom_allreduce_(
                reduced,
                routed,
                tp_rank_,
                static_cast<int>(tp_size_));
            if (reduced_by_custom) {
                routed = reduced;
            }
        }
        if (!reduced_by_custom) {
            if (moe_allreduce_outplace_enabled_) {
                auto reduced = allreduce_scratch_.get(
                    routed->shape(),
                    routed->dtype(),
                    routed->device());
                infinicore::op::distributed::allreduce_(reduced, routed, INFINICCL_SUM, communicator_);
                routed = reduced;
            } else {
                infinicore::op::distributed::allreduce_(routed, routed, INFINICCL_SUM, communicator_);
            }
        }
        debug_dump_tensor(routed, layer_idx_, "after_allreduce", debug_dump_enabled_);
    }
    debug_dump_tensor(routed, layer_idx_, "moe_output", debug_dump_enabled_);
    return routed;
}

void DeepseekV4MoE::process_weights_after_loading() {
    experts_->process_weights_after_loading();
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
