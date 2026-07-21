#include "deepseek_v4_mlp.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/moe/ep/ep_config.hpp"
#include "deepseek_v4_profile.hpp"

#include "infinicore/ops/add.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/deepseek_v4_biased_topk.hpp"
#include "infinicore/ops/deepseek_v4_dynamic_scaled_int8_quant.hpp"
#include "infinicore/ops/deepseek_v4_hash_topk.hpp"
#include "infinicore/ops/deepseek_v4_lightop_moe_marlin.hpp"
#include "infinicore/ops/deepseek_v4_linear_bf16_fp32.hpp"
#include "infinicore/ops/deepseek_v4_mhc.hpp"
#include "infinicore/ops/deepseek_v4_moe_align_block_size.hpp"
#include "infinicore/ops/deepseek_v4_moe_lmslim_marlin_w8a8.hpp"
#include "infinicore/ops/deepseek_v4_moe_marlin_repack.hpp"
#include "infinicore/ops/deepseek_v4_moe_marlin_w8a8.hpp"
#include "infinicore/ops/deepseek_v4_moe_sum.hpp"
#include "infinicore/ops/deepseek_v4_silu_and_mul.hpp"
#include "infinicore/ops/deepseek_v4_silu_and_mul_clamp.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/mul_scalar.hpp"

#include <algorithm>
#include <cstdlib>
#include <infiniccl.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {

namespace {

bool env_flag_enabled(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON" || text == "marlin";
}

int env_int_or(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::stoi(value);
}

bool debug_dump_enabled() {
    return env_flag_enabled("INFINILM_DSV4_DEBUG_DUMP");
}

void debug_dump_tensor(const infinicore::Tensor &tensor, size_t layer_idx, const std::string &name) {
    if (!debug_dump_enabled() || !tensor) {
        return;
    }
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tensor->debug("/tmp/infinilm_dsv4_tp" + std::to_string(rank_info.tp_rank) + "_l" + std::to_string(layer_idx) + "_" + name + ".bin");
}

struct MarlinGemmConfig {
    int block_size;
    int gemm1_mode;
    int gemm2_mode;
    int delta;
    bool supported;
};

MarlinGemmConfig select_marlin_gemm_config(size_t num_tokens,
                                           size_t hidden_size,
                                           size_t intermediate_size,
                                           size_t top_k,
                                           int fallback_block_size,
                                           int fallback_mode,
                                           int fallback_delta) {
    MarlinGemmConfig config{fallback_block_size, fallback_mode, fallback_mode, fallback_delta, false};

    if (hidden_size == 7168 && intermediate_size == 256 && top_k == 8) {
        config.block_size = 16;
        config.delta = 1;
        config.supported = true;
        if (num_tokens <= 1) {
            config.gemm1_mode = 21;
            config.gemm2_mode = 25;
        } else if (num_tokens <= 7) {
            config.gemm1_mode = 78;
            config.gemm2_mode = 73;
        } else if (num_tokens <= 16) {
            config.gemm1_mode = 29;
            config.gemm2_mode = 12;
        } else if (num_tokens <= 75) {
            config.gemm1_mode = 55;
            config.gemm2_mode = 54;
        }
    }
    if (hidden_size == 4096 && intermediate_size == 256 && top_k == 6) {
        config.block_size = 16;
        config.delta = 1;
        config.supported = true;
        if (num_tokens <= 1) {
            config.gemm1_mode = 58;
            config.gemm2_mode = 16;
        } else if (num_tokens <= 7) {
            config.gemm1_mode = 29;
            config.gemm2_mode = 9;
        } else if (num_tokens <= 16) {
            config.gemm1_mode = 29;
            config.gemm2_mode = 4;
        } else if (num_tokens <= 75) {
            config.gemm1_mode = 29;
            config.gemm2_mode = 55;
        } else {
            config.gemm1_mode = 37;
            config.gemm2_mode = 54;
        }
    }

    const int block_override = env_int_or("INFINILM_DSV4_MOE_MARLIN_BLOCK_SIZE", 0);
    const int mode_override = env_int_or("INFINILM_DSV4_MOE_MARLIN_MODE", -1);
    const int mode1_override = env_int_or("INFINILM_DSV4_MOE_MARLIN_MODE1", -1);
    const int mode2_override = env_int_or("INFINILM_DSV4_MOE_MARLIN_MODE2", -1);
    const int delta_override = env_int_or("INFINILM_DSV4_MOE_MARLIN_DELTA", -1);
    if (block_override > 0) {
        config.block_size = block_override;
    }
    if (mode_override >= 0) {
        config.gemm1_mode = mode_override;
        config.gemm2_mode = mode_override;
        config.supported = true;
    }
    if (mode1_override >= 0) {
        config.gemm1_mode = mode1_override;
        config.supported = true;
    }
    if (mode2_override >= 0) {
        config.gemm2_mode = mode2_override;
        config.supported = true;
    }
    if (delta_override >= 0) {
        config.delta = delta_override;
    }
    return config;
}

} // namespace

DeepseekV4MoEGate::DeepseekV4MoEGate(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t layer_idx,
                                     const infinicore::Device &device) {
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_experts = model_config->get_or_alias<size_t>("n_routed_experts", "num_experts", 0);
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t num_experts_per_tok = model_config->get<size_t>("num_experts_per_tok");
    const size_t num_hash_layers = model_config->get_or<size_t>("num_hash_layers", 0);
    num_experts_per_tok_ = num_experts_per_tok;
    norm_topk_prob_ = model_config->get_or<bool>("norm_topk_prob", true);
    is_hash_ = layer_idx < num_hash_layers;
    if (num_experts == 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4MoEGate: n_routed_experts is required");
    }
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts, hidden_size}, model_config->get_dtype(), device));
    if (is_hash_) {
        INFINICORE_NN_PARAMETER_INIT(tid2eid, ({vocab_size, num_experts_per_tok}, infinicore::DataType::I64, device));
    } else {
        INFINICORE_NN_PARAMETER_INIT(bias, ({num_experts}, infinicore::DataType::F32, device));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
DeepseekV4MoEGate::forward(const infinicore::Tensor &hidden_states,
                           const infinicore::Tensor &input_ids) const {
    if (hidden_states->ndim() != 2) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4MoEGate::forward expects hidden_states [tokens, hidden]");
    }
    auto router_logits = infinicore::op::deepseek_v4_linear_bf16_fp32(hidden_states, weight_);
    auto router_scores = infinicore::Tensor::empty(
        {hidden_states->size(0), num_experts_per_tok_},
        infinicore::DataType::F32,
        hidden_states->device());
    auto router_indices = infinicore::Tensor::empty(
        {hidden_states->size(0), num_experts_per_tok_},
        infinicore::DataType::I32,
        hidden_states->device());
    if (is_hash_) {
        infinicore::op::deepseek_v4_hash_topk_naive_(
            router_scores,
            router_indices,
            router_logits,
            input_ids,
            tid2eid_,
            norm_topk_prob_);
    } else {
        infinicore::op::deepseek_v4_topk_naive_(
            router_scores,
            router_indices,
            router_logits,
            bias_,
            norm_topk_prob_);
    }
    return {router_scores, router_indices};
}

DeepseekV4SharedExperts::DeepseekV4SharedExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    const auto quantization_method = model_config->get_quantization_method();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size") * model_config->get_or<size_t>("n_shared_experts", 1);
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    if (tp_size == 0 || intermediate_size % tp_size != 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4SharedExperts: intermediate_size must be divisible by tp_size");
    }

    w1_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "w1", hidden_size, intermediate_size, quantization_method, false, dtype, device, tp_rank, tp_size);
    w3_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "w3", hidden_size, intermediate_size, quantization_method, false, dtype, device, tp_rank, tp_size);
    w2_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "w2", intermediate_size, hidden_size, quantization_method, false, dtype, device, tp_rank, tp_size, nullptr);
}

infinicore::Tensor DeepseekV4SharedExperts::forward(infinicore::Tensor hidden_states) const {
    auto gate = w1_->forward(hidden_states);
    auto up = w3_->forward(hidden_states);
    auto gate_up = infinicore::op::cat({gate, up}, -1);
    auto activated = infinicore::op::deepseek_v4_silu_and_mul(gate_up);
    return w2_->forward(activated);
}

void DeepseekV4SharedExperts::process_weights_after_loading() {
    w1_->process_weights_after_loading();
    w2_->process_weights_after_loading();
    w3_->process_weights_after_loading();
}

void DeepseekV4SharedExperts::reset_runtime_state() const {
    w1_->reset_runtime_state();
    w2_->reset_runtime_state();
    w3_->reset_runtime_state();
}

DeepseekV4PackedExperts::DeepseekV4PackedExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 const infinicore::Device &device) {
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
    use_marlin_backend_ = env_flag_enabled("INFINILM_DSV4_MOE_MARLIN") || env_flag_enabled("INFINILM_DSV4_MOE_BACKEND");
    marlin_block_size_ = env_int_or("INFINILM_DSV4_MOE_MARLIN_BLOCK_SIZE", 16);
    marlin_mode_ = env_int_or("INFINILM_DSV4_MOE_MARLIN_MODE", 54);
    marlin_delta_ = env_int_or("INFINILM_DSV4_MOE_MARLIN_DELTA", 1);

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

bool DeepseekV4PackedExperts::use_marlin_backend() const {
    return use_marlin_backend_;
}

void DeepseekV4PackedExperts::process_weights_after_loading() {
    if (!use_marlin_backend_) {
        return;
    }
    const auto process_config = select_marlin_gemm_config(1, hidden_size_, intermediate_size_per_partition_, num_experts_per_tok_, marlin_block_size_, marlin_mode_, marlin_delta_);
    if (!process_config.supported) {
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
}

infinicore::Tensor DeepseekV4PackedExperts::forward_reference(const infinicore::Tensor &hidden_states,
                                                              const infinicore::Tensor &topk_weights,
                                                              const infinicore::Tensor &topk_indices) const {
    auto output = infinicore::Tensor::empty(hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
    infinicore::op::deepseek_v4_moe_w8a8_naive_(
        output,
        hidden_states,
        topk_weights,
        topk_indices,
        w13_weight_,
        w13_weight_scale_,
        w2_weight_,
        w2_weight_scale_,
        10.0);
    if (routed_scaling_factor_ != 1.0) {
        output = infinicore::op::mul_scalar(output, routed_scaling_factor_);
    }
    return output;
}

infinicore::Tensor DeepseekV4PackedExperts::forward_marlin(const infinicore::Tensor &hidden_states,
                                                           const infinicore::Tensor &topk_weights,
                                                           const infinicore::Tensor &topk_indices) const {
    if (hidden_states->ndim() != 2 || topk_indices->ndim() != 2 || topk_weights->ndim() != 2) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4PackedExperts::forward_marlin expects hidden [tokens, hidden] and topk [tokens, topk]");
    }
    const size_t num_tokens = hidden_states->size(0);
    const size_t top_k = topk_indices->size(1);
    const size_t flat_topk = num_tokens * top_k;
    const size_t gate_up_size = intermediate_size_per_partition_ * 2;
    const auto marlin_config = select_marlin_gemm_config(
        num_tokens,
        hidden_size_,
        intermediate_size_per_partition_,
        top_k,
        marlin_block_size_,
        marlin_mode_,
        marlin_delta_);
    if (!marlin_config.supported) {
        return forward_reference(hidden_states, topk_weights, topk_indices);
    }
    const int op_num_experts = static_cast<int>(num_experts_);
    const int block_size = marlin_config.block_size;
    size_t max_num_tokens_padded = flat_topk + static_cast<size_t>(op_num_experts) * static_cast<size_t>(block_size - 1);
    max_num_tokens_padded = ((max_num_tokens_padded + static_cast<size_t>(block_size - 1)) / static_cast<size_t>(block_size)) * static_cast<size_t>(block_size);

    auto sorted_token_ids = infinicore::Tensor::empty(
        {max_num_tokens_padded}, infinicore::DataType::I32, hidden_states->device());
    auto expert_ids = infinicore::Tensor::empty(
        {(max_num_tokens_padded + static_cast<size_t>(block_size - 1)) / static_cast<size_t>(block_size)},
        infinicore::DataType::I32,
        hidden_states->device());
    auto num_tokens_post_pad = infinicore::Tensor::empty({1}, infinicore::DataType::I32, hidden_states->device());
    auto cumsum_buffer = infinicore::Tensor::empty(
        {static_cast<size_t>(op_num_experts + 1)}, infinicore::DataType::I32, hidden_states->device());
    infinicore::op::deepseek_v4_moe_align_block_size_(
        topk_indices,
        op_num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        true);

    auto q_hidden = infinicore::Tensor::empty(hidden_states->shape(), infinicore::DataType::I8, hidden_states->device());
    auto hidden_scale = infinicore::Tensor::empty({num_tokens, 1}, infinicore::DataType::F32, hidden_states->device());
    infinicore::op::deepseek_v4_dynamic_scaled_int8_quant_(q_hidden, hidden_states, hidden_scale, std::nullopt);

    if (!w13_weight_marlin_ || !w2_weight_marlin_) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4PackedExperts::forward_marlin requires process_weights_after_loading() before forward");
    }

    const bool use_aiter_marlin = env_flag_enabled("INFINILM_DSV4_MOE_AITER");
    const bool use_split_lightop = env_flag_enabled("INFINILM_DSV4_MOE_SPLIT_LIGHTOP");
    if (!use_aiter_marlin && !use_split_lightop) {
        auto output = infinicore::Tensor::empty(hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
        infinicore::op::deepseek_v4_moe_lmslim_marlin_w8a8_(
            output,
            hidden_states->contiguous(),
            w13_weight_marlin_,
            w2_weight_marlin_,
            topk_weights,
            topk_indices,
            w13_weight_scale_,
            w2_weight_scale_,
            static_cast<int64_t>(num_experts_),
            routed_scaling_factor_);
        return output;
    }

    auto gate_up = infinicore::Tensor::empty(
        {num_tokens, top_k, gate_up_size}, hidden_states->dtype(), hidden_states->device());
    if (use_aiter_marlin) {
        infinicore::op::deepseek_v4_moe_marlin_w8a8_(
            q_hidden,
            w13_weight_marlin_,
            gate_up,
            hidden_scale,
            w13_weight_scale_,
            std::nullopt,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            static_cast<int>(top_k),
            marlin_config.gemm1_mode,
            marlin_config.delta);
    } else {
        infinicore::op::deepseek_v4_lightop_moe_gemm_marlin_w8a8_(
            q_hidden,
            w13_weight_marlin_,
            gate_up,
            hidden_scale,
            w13_weight_scale_,
            std::nullopt,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            static_cast<int>(top_k),
            marlin_config.gemm1_mode,
            marlin_config.delta);
    }

    auto q_activated = infinicore::Tensor::empty(
        {flat_topk, intermediate_size_per_partition_}, infinicore::DataType::I8, hidden_states->device());
    auto activated_scale = infinicore::Tensor::empty({flat_topk, 1}, infinicore::DataType::F32, hidden_states->device());
    if (use_aiter_marlin) {
        auto activated = infinicore::op::deepseek_v4_silu_and_mul(
            gate_up->view({flat_topk, gate_up_size}));
        infinicore::op::deepseek_v4_dynamic_scaled_int8_quant_(
            q_activated,
            activated,
            activated_scale,
            std::nullopt);
    } else {
        infinicore::op::deepseek_v4_lightop_fuse_silu_mul_quant_(
            q_activated,
            activated_scale,
            gate_up->view({flat_topk, gate_up_size}),
            std::nullopt,
            1,
            -1,
            std::nullopt);
    }

    auto down = infinicore::Tensor::empty(
        {num_tokens, top_k, hidden_size_}, hidden_states->dtype(), hidden_states->device());
    if (use_aiter_marlin) {
        infinicore::op::deepseek_v4_moe_marlin_w8a8_(
            q_activated,
            w2_weight_marlin_,
            down,
            activated_scale,
            w2_weight_scale_,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            1,
            marlin_config.gemm2_mode,
            marlin_config.delta);
    } else {
        infinicore::op::deepseek_v4_lightop_moe_gemm_marlin_w8a8_(
            q_activated,
            w2_weight_marlin_,
            down,
            activated_scale,
            w2_weight_scale_,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            1,
            marlin_config.gemm2_mode,
            marlin_config.delta);
    }

    auto output = infinicore::Tensor::empty(hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
    if (use_aiter_marlin) {
        infinicore::op::deepseek_v4_moe_sum_(output, down);
        if (routed_scaling_factor_ != 1.0) {
            output = infinicore::op::mul_scalar(output, routed_scaling_factor_);
        }
    } else {
        infinicore::op::deepseek_v4_lightop_moe_sum_(
            output,
            down,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            static_cast<float>(routed_scaling_factor_),
            -1);
    }
    return output;
}

infinicore::Tensor DeepseekV4PackedExperts::forward(const infinicore::Tensor &hidden_states,
                                                    const infinicore::Tensor &topk_weights,
                                                    const infinicore::Tensor &topk_indices) const {
    if (use_marlin_backend()) {
        return forward_marlin(hidden_states, topk_weights, topk_indices);
    }
    return forward_reference(hidden_states, topk_weights, topk_indices);
}

DeepseekV4MoE::DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             size_t layer_idx,
                             const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
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
    debug_dump_tensor(hidden_states, layer_idx_, "moe_input");
    infinicore::Tensor routing_weights;
    infinicore::Tensor selected_experts;
    {
        profile::ScopedTimer timer(profile::Event::MoeTopk, token_count);
        std::tie(routing_weights, selected_experts) = gate_->forward(hidden_states, input_ids);
    }
    debug_dump_tensor(routing_weights, layer_idx_, "topk_weights");
    debug_dump_tensor(selected_experts, layer_idx_, "topk_indices");

    infinicore::Tensor routed;
    {
        profile::ScopedTimer timer(profile::Event::MoeExperts, token_count);
        routed = experts_->forward(hidden_states, routing_weights, selected_experts);
    }
    debug_dump_tensor(routed, layer_idx_, "routed");

    if (shared_experts_) {
        infinicore::Tensor shared;
        {
            profile::ScopedTimer timer(profile::Event::MoeSharedExperts, token_count);
            shared = shared_experts_->forward(hidden_states);
        }
        debug_dump_tensor(shared, layer_idx_, "shared");
        {
            profile::ScopedTimer timer(profile::Event::MoeAddShared, token_count);
            routed = infinicore::op::add(routed, shared);
        }
        debug_dump_tensor(routed, layer_idx_, "after_shared");
    }
    if (tp_size_ > 1 && communicator_ != nullptr) {
        profile::ScopedTimer timer(profile::Event::MoeAllReduce, token_count);
        infinicore::op::distributed::allreduce_(routed, routed, INFINICCL_SUM, communicator_);
        debug_dump_tensor(routed, layer_idx_, "after_allreduce");
    }
    debug_dump_tensor(routed, layer_idx_, "moe_output");
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
