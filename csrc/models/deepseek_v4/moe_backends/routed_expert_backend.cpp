#include "routed_expert_backend.hpp"

#include "../deepseek_v4_profile.hpp"
#include "../deepseek_v4_utils.hpp"
#include "infinicore/ops/deepseek_v4_dynamic_scaled_int8_quant.hpp"
#include "infinicore/ops/deepseek_v4_fused_experts_impl_int8_marlin.hpp"
#include "infinicore/ops/deepseek_v4_lightop_moe_marlin.hpp"
#include "infinicore/ops/deepseek_v4_moe_align_block_size.hpp"
#include "infinicore/ops/deepseek_v4_moe_lmslim_marlin_w8a8.hpp"
#include "infinicore/ops/deepseek_v4_moe_marlin_w8a8.hpp"
#include "infinicore/ops/deepseek_v4_moe_sum.hpp"
#include "infinicore/ops/deepseek_v4_moe_w8a8.hpp"
#include "infinicore/ops/deepseek_v4_silu_and_mul.hpp"
#include "infinicore/ops/mul_scalar.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>

namespace infinilm::models::deepseek_v4::moe_backends {

namespace {

RoutedExpertBackend parse_backend(const std::string &text) {
    if (text == "reference" || text == "naive") {
        return RoutedExpertBackend::Naive;
    }
    if (text == "lmslim" || text == "lmslim_fused" || text == "fused") {
        return RoutedExpertBackend::LmslimFused;
    }
    if (text == "fused_experts_int8_marlin" || text == "int8_marlin" || text == "sglang_int8_marlin") {
        return RoutedExpertBackend::FusedExpertsInt8Marlin;
    }
    if (text == "aiter" || text == "aiter_split") {
        return RoutedExpertBackend::AiterSplit;
    }
    if (text == "lightop" || text == "lightop_split" || text == "split_lightop") {
        return RoutedExpertBackend::LightopSplit;
    }
    throw std::runtime_error("INFINILM_DSV4_ROUTED_EXPERT_BACKEND must be one of: naive, lmslim_fused, fused_experts_int8_marlin, aiter_split, lightop_split");
}

infinicore::Tensor forward_naive(const RoutedExpertContext &ctx,
                                 DeepseekV4RoutedExpertScratch &scratch,
                                 const infinicore::Tensor &hidden_states,
                                 const infinicore::Tensor &topk_weights,
                                 const infinicore::Tensor &topk_indices) {
    if (!ctx.w13_weight || !ctx.w2_weight) {
        throw std::runtime_error("DeepseekV4 routed expert naive backend requires original packed weights, but they were released after Marlin repack");
    }
    auto output = scratch.get_output(
        {hidden_states->size(0), ctx.hidden_size},
        hidden_states->dtype(),
        hidden_states->device());
    infinicore::op::deepseek_v4_moe_w8a8_(
        output,
        hidden_states,
        topk_weights,
        topk_indices,
        ctx.w13_weight,
        ctx.w13_weight_scale,
        ctx.w2_weight,
        ctx.w2_weight_scale,
        10.0);

    // sglang代码中，当检测到hip硬件时，会跳过下面的代码。
    // if (ctx.routed_scaling_factor != 1.0) {
    //     output = infinicore::op::mul_scalar(output, ctx.routed_scaling_factor);
    // }

    return output;
}

void require_marlin_weights(RoutedExpertBackend backend, const RoutedExpertContext &ctx) {
    if (!ctx.w13_weight_marlin || !ctx.w2_weight_marlin) {
        throw std::runtime_error(std::string("DeepseekV4 routed expert backend ") + to_string(backend) + " requires process_weights_after_loading() to prepare Marlin weights");
    }
}

struct MarlinPreparedInputs {
    size_t num_tokens{0};
    size_t top_k{0};
    size_t flat_topk{0};
    size_t gate_up_size{0};
    MarlinGemmConfig config;
    infinicore::Tensor sorted_token_ids;
    infinicore::Tensor expert_ids;
    infinicore::Tensor num_tokens_post_pad;
    infinicore::Tensor q_hidden;
    infinicore::Tensor hidden_scale;
};

MarlinPreparedInputs prepare_marlin_inputs(RoutedExpertBackendChoice choice,
                                           const RoutedExpertContext &ctx,
                                           const infinicore::Tensor &hidden_states,
                                           const infinicore::Tensor &topk_indices) {
    const size_t profile_tokens = hidden_states->ndim() > 0 ? hidden_states->size(0) : 0;
    profile::ScopedTimer prepare_timer(profile::Event::MoeExpertsPrepare, profile_tokens);
    if (hidden_states->ndim() != 2 || topk_indices->ndim() != 2) {
        throw std::runtime_error("DeepseekV4 routed expert Marlin backend expects hidden [tokens, hidden] and topk [tokens, topk]");
    }
    MarlinPreparedInputs prep;
    prep.num_tokens = hidden_states->size(0);
    prep.top_k = topk_indices->size(1);
    prep.flat_topk = prep.num_tokens * prep.top_k;
    prep.gate_up_size = ctx.intermediate_size_per_partition * 2;
    prep.config = select_marlin_gemm_config(
        prep.num_tokens,
        ctx.hidden_size,
        ctx.intermediate_size_per_partition,
        prep.top_k,
        ctx.marlin_block_size,
        ctx.marlin_mode,
        ctx.marlin_delta,
        ctx.marlin_override);
    if (!prep.config.supported) {
        if (choice.explicit_backend) {
            throw std::runtime_error(std::string("DeepseekV4 routed expert backend ") + to_string(choice.backend) + " does not support this token/shape combination");
        }
        return prep;
    }

    require_marlin_weights(choice.backend, ctx);
    const int op_num_experts = static_cast<int>(ctx.num_experts);
    const int block_size = prep.config.block_size;
    size_t max_num_tokens_padded = prep.flat_topk + static_cast<size_t>(op_num_experts) * static_cast<size_t>(block_size - 1);
    max_num_tokens_padded = ((max_num_tokens_padded + static_cast<size_t>(block_size - 1)) / static_cast<size_t>(block_size)) * static_cast<size_t>(block_size);
    if (prep.flat_topk < static_cast<size_t>(op_num_experts)) {
        max_num_tokens_padded = std::min(prep.flat_topk * static_cast<size_t>(block_size), max_num_tokens_padded);
    }

    prep.sorted_token_ids = infinicore::Tensor::empty(
        {max_num_tokens_padded}, infinicore::DataType::I32, hidden_states->device());
    prep.expert_ids = infinicore::Tensor::empty(
        {(max_num_tokens_padded + static_cast<size_t>(block_size - 1)) / static_cast<size_t>(block_size)},
        infinicore::DataType::I32,
        hidden_states->device());
    prep.num_tokens_post_pad = infinicore::Tensor::empty({1}, infinicore::DataType::I32, hidden_states->device());
    auto cumsum_buffer = infinicore::Tensor::empty(
        {static_cast<size_t>(op_num_experts + 1)}, infinicore::DataType::I32, hidden_states->device());
    {
        profile::ScopedTimer timer(profile::Event::MoeExpertsPrepareAlign, prep.num_tokens);
        infinicore::op::deepseek_v4_moe_align_block_size_(
            topk_indices,
            op_num_experts,
            block_size,
            prep.sorted_token_ids,
            prep.expert_ids,
            prep.num_tokens_post_pad,
            cumsum_buffer,
            true);
    }

    prep.q_hidden = infinicore::Tensor::empty(hidden_states->shape(), infinicore::DataType::I8, hidden_states->device());
    prep.hidden_scale = infinicore::Tensor::empty({prep.num_tokens, 1}, infinicore::DataType::F32, hidden_states->device());
    {
        profile::ScopedTimer timer(profile::Event::MoeExpertsPrepareQuant, prep.num_tokens);
        infinicore::op::deepseek_v4_dynamic_scaled_int8_quant_(prep.q_hidden, hidden_states, prep.hidden_scale, std::nullopt);
    }
    return prep;
}

infinicore::Tensor forward_lmslim_fused(const RoutedExpertContext &ctx,
                                        const infinicore::Tensor &hidden_states,
                                        const infinicore::Tensor &topk_weights,
                                        const infinicore::Tensor &topk_indices) {
    require_marlin_weights(RoutedExpertBackend::LmslimFused, ctx);
    infinicore::Tensor contiguous_hidden;
    {
        profile::ScopedTimer timer(profile::Event::MoeExpertsContiguous, hidden_states->size(0));
        contiguous_hidden = hidden_states->contiguous();
    }
    {
        profile::ScopedTimer timer(profile::Event::MoeExpertsFusedCall, hidden_states->size(0));
        infinicore::op::deepseek_v4_fused_experts_impl_int8_marlin_(
            contiguous_hidden,
            contiguous_hidden,
            ctx.w13_weight_marlin,
            ctx.w2_weight_marlin,
            topk_weights,
            topk_indices,
            ctx.w13_weight_scale,
            ctx.w2_weight_scale,
            static_cast<int64_t>(ctx.num_experts),
            ctx.routed_scaling_factor,
            true);
    }
    return contiguous_hidden;
}

infinicore::Tensor forward_fused_experts_int8_marlin(const RoutedExpertContext &ctx,
                                                     DeepseekV4RoutedExpertScratch &scratch,
                                                     const infinicore::Tensor &hidden_states,
                                                     const infinicore::Tensor &topk_weights,
                                                     const infinicore::Tensor &topk_indices,
                                                     const std::optional<infinicore::Tensor> &shared_output) {
    require_marlin_weights(RoutedExpertBackend::FusedExpertsInt8Marlin, ctx);
    infinicore::Tensor contiguous_hidden;
    {
        profile::ScopedTimer timer(profile::Event::MoeExpertsContiguous, hidden_states->size(0));
        contiguous_hidden = scratch.get_contiguous_hidden(
            {hidden_states->size(0), ctx.hidden_size},
            hidden_states->dtype(),
            hidden_states->device());
        // contiguous_hidden->copy_from(hidden_states);
    }
    {
        profile::ScopedTimer timer(profile::Event::MoeExpertsFusedCall, hidden_states->size(0));
        infinicore::op::deepseek_v4_fused_experts_impl_int8_marlin_(
            contiguous_hidden,
            hidden_states,
            ctx.w13_weight_marlin,
            ctx.w2_weight_marlin,
            topk_weights,
            topk_indices,
            ctx.w13_weight_scale,
            ctx.w2_weight_scale,
            static_cast<int64_t>(ctx.num_experts),
            ctx.routed_scaling_factor,
            true,
            shared_output);
    }
    return contiguous_hidden;
}

infinicore::Tensor forward_aiter_split(const RoutedExpertContext &ctx,
                                       const MarlinPreparedInputs &prep,
                                       const infinicore::Tensor &hidden_states,
                                       const infinicore::Tensor &topk_weights) {
    auto gate_up = infinicore::Tensor::empty(
        {prep.num_tokens, prep.top_k, prep.gate_up_size}, hidden_states->dtype(), hidden_states->device());
    infinicore::op::deepseek_v4_moe_marlin_w8a8_(
        prep.q_hidden,
        ctx.w13_weight_marlin,
        gate_up,
        prep.hidden_scale,
        ctx.w13_weight_scale,
        std::nullopt,
        prep.sorted_token_ids,
        prep.expert_ids,
        prep.num_tokens_post_pad,
        static_cast<int>(prep.top_k),
        prep.config.gemm1_mode,
        prep.config.delta);

    auto q_activated = infinicore::Tensor::empty(
        {prep.flat_topk, ctx.intermediate_size_per_partition}, infinicore::DataType::I8, hidden_states->device());
    auto activated_scale = infinicore::Tensor::empty({prep.flat_topk, 1}, infinicore::DataType::F32, hidden_states->device());
    auto activated = infinicore::op::deepseek_v4_silu_and_mul(
        gate_up->view({prep.flat_topk, prep.gate_up_size}));
    infinicore::op::deepseek_v4_dynamic_scaled_int8_quant_(
        q_activated,
        activated,
        activated_scale,
        std::nullopt);

    auto down = infinicore::Tensor::empty(
        {prep.num_tokens, prep.top_k, ctx.hidden_size}, hidden_states->dtype(), hidden_states->device());
    infinicore::op::deepseek_v4_moe_marlin_w8a8_(
        q_activated,
        ctx.w2_weight_marlin,
        down,
        activated_scale,
        ctx.w2_weight_scale,
        topk_weights,
        prep.sorted_token_ids,
        prep.expert_ids,
        prep.num_tokens_post_pad,
        1,
        prep.config.gemm2_mode,
        prep.config.delta);

    auto output = infinicore::Tensor::empty(hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
    infinicore::op::deepseek_v4_moe_sum_(output, down);
    if (ctx.routed_scaling_factor != 1.0) {
        output = infinicore::op::mul_scalar(output, ctx.routed_scaling_factor);
    }
    return output;
}

infinicore::Tensor forward_lightop_split(const RoutedExpertContext &ctx,
                                         const MarlinPreparedInputs &prep,
                                         const infinicore::Tensor &hidden_states,
                                         const infinicore::Tensor &topk_weights) {
    auto gate_up = infinicore::Tensor::empty(
        {prep.num_tokens, prep.top_k, prep.gate_up_size}, hidden_states->dtype(), hidden_states->device());
    infinicore::op::deepseek_v4_lightop_moe_gemm_marlin_w8a8_(
        prep.q_hidden,
        ctx.w13_weight_marlin,
        gate_up,
        prep.hidden_scale,
        ctx.w13_weight_scale,
        std::nullopt,
        prep.sorted_token_ids,
        prep.expert_ids,
        prep.num_tokens_post_pad,
        static_cast<int>(prep.top_k),
        prep.config.gemm1_mode,
        prep.config.delta);

    auto q_activated = infinicore::Tensor::empty(
        {prep.flat_topk, ctx.intermediate_size_per_partition}, infinicore::DataType::I8, hidden_states->device());
    auto activated_scale = infinicore::Tensor::empty({prep.flat_topk, 1}, infinicore::DataType::F32, hidden_states->device());
    infinicore::op::deepseek_v4_lightop_fuse_silu_mul_quant_(
        q_activated,
        activated_scale,
        gate_up->view({prep.flat_topk, prep.gate_up_size}),
        std::nullopt,
        1,
        -1,
        std::nullopt);

    auto down = infinicore::Tensor::empty(
        {prep.num_tokens, prep.top_k, ctx.hidden_size}, hidden_states->dtype(), hidden_states->device());
    infinicore::op::deepseek_v4_lightop_moe_gemm_marlin_w8a8_(
        q_activated,
        ctx.w2_weight_marlin,
        down,
        activated_scale,
        ctx.w2_weight_scale,
        topk_weights,
        prep.sorted_token_ids,
        prep.expert_ids,
        prep.num_tokens_post_pad,
        1,
        prep.config.gemm2_mode,
        prep.config.delta);

    auto output = infinicore::Tensor::empty(hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
    infinicore::op::deepseek_v4_lightop_moe_sum_(
        output,
        down,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        static_cast<float>(ctx.routed_scaling_factor),
        -1);
    return output;
}

} // namespace

RoutedExpertBackendChoice select_routed_expert_backend() {
    const char *backend_value = utils::env_value("INFINILM_DSV4_ROUTED_EXPERT_BACKEND");
    if (backend_value != nullptr && backend_value[0] != '\0') {
        return {parse_backend(backend_value), true};
    }
    return {RoutedExpertBackend::FusedExpertsInt8Marlin, false};
}

const char *to_string(RoutedExpertBackend backend) {
    switch (backend) {
    case RoutedExpertBackend::Naive:
        return "naive";
    case RoutedExpertBackend::LmslimFused:
        return "lmslim_fused";
    case RoutedExpertBackend::FusedExpertsInt8Marlin:
        return "fused_experts_int8_marlin";
    case RoutedExpertBackend::AiterSplit:
        return "aiter_split";
    case RoutedExpertBackend::LightopSplit:
        return "lightop_split";
    }
    return "unknown";
}

bool requires_marlin_repack(RoutedExpertBackend backend) {
    return backend != RoutedExpertBackend::Naive;
}

MarlinGemmOverride read_marlin_gemm_override_from_env() {
    return {
        utils::env_int_or("INFINILM_DSV4_MOE_MARLIN_BLOCK_SIZE", 0),
        utils::env_int_or("INFINILM_DSV4_MOE_MARLIN_MODE", -1),
        utils::env_int_or("INFINILM_DSV4_MOE_MARLIN_MODE1", -1),
        utils::env_int_or("INFINILM_DSV4_MOE_MARLIN_MODE2", -1),
        utils::env_int_or("INFINILM_DSV4_MOE_MARLIN_DELTA", -1),
    };
}

MarlinGemmConfig select_marlin_gemm_config(size_t num_tokens,
                                           size_t hidden_size,
                                           size_t intermediate_size,
                                           size_t top_k,
                                           int fallback_block_size,
                                           int fallback_mode,
                                           int fallback_delta,
                                           const MarlinGemmOverride &override_config) {
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
            config.gemm2_mode = 12;
        } else {
            config.gemm1_mode = 37;
            config.gemm2_mode = 54;
        }
    }

    if (override_config.block_size > 0) {
        config.block_size = override_config.block_size;
    }
    if (override_config.mode >= 0) {
        config.gemm1_mode = override_config.mode;
        config.gemm2_mode = override_config.mode;
        config.supported = true;
    }
    if (override_config.gemm1_mode >= 0) {
        config.gemm1_mode = override_config.gemm1_mode;
        config.supported = true;
    }
    if (override_config.gemm2_mode >= 0) {
        config.gemm2_mode = override_config.gemm2_mode;
        config.supported = true;
    }
    if (override_config.delta >= 0) {
        config.delta = override_config.delta;
    }
    return config;
}

infinicore::Tensor forward_routed_experts(RoutedExpertBackendChoice choice,
                                          const RoutedExpertContext &ctx,
                                          DeepseekV4RoutedExpertScratch &scratch,
                                          const infinicore::Tensor &hidden_states,
                                          const infinicore::Tensor &topk_weights,
                                          const infinicore::Tensor &topk_indices,
                                          const std::optional<infinicore::Tensor> &shared_output) {
    if (choice.backend == RoutedExpertBackend::Naive) {
        return forward_naive(ctx, scratch, hidden_states, topk_weights, topk_indices);
    }

    const auto config = select_marlin_gemm_config(
        hidden_states->size(0),
        ctx.hidden_size,
        ctx.intermediate_size_per_partition,
        topk_indices->size(1),
        ctx.marlin_block_size,
        ctx.marlin_mode,
        ctx.marlin_delta,
        ctx.marlin_override);
    if (!config.supported) {
        if (ctx.raw_weights_released) {
            throw std::runtime_error(std::string("DeepseekV4 routed expert backend ") + to_string(choice.backend) + " does not support this token shape after original weights were released");
        }
        return forward_naive(ctx, scratch, hidden_states, topk_weights, topk_indices);
    }

    switch (choice.backend) {
    case RoutedExpertBackend::LmslimFused:
        return forward_lmslim_fused(ctx, hidden_states, topk_weights, topk_indices);
    case RoutedExpertBackend::FusedExpertsInt8Marlin:
        return forward_fused_experts_int8_marlin(ctx, scratch, hidden_states, topk_weights, topk_indices, shared_output);
    case RoutedExpertBackend::AiterSplit:
    case RoutedExpertBackend::LightopSplit:
        break;
    case RoutedExpertBackend::Naive:
        return forward_naive(ctx, scratch, hidden_states, topk_weights, topk_indices);
    }

    switch (choice.backend) {
    case RoutedExpertBackend::AiterSplit: {
        auto prep = prepare_marlin_inputs(choice, ctx, hidden_states, topk_indices);
        return forward_aiter_split(ctx, prep, hidden_states, topk_weights);
    }
    case RoutedExpertBackend::LightopSplit: {
        auto prep = prepare_marlin_inputs(choice, ctx, hidden_states, topk_indices);
        return forward_lightop_split(ctx, prep, hidden_states, topk_weights);
    }
    case RoutedExpertBackend::LmslimFused:
    case RoutedExpertBackend::FusedExpertsInt8Marlin:
    case RoutedExpertBackend::Naive:
        break;
    }
    return forward_naive(ctx, scratch, hidden_states, topk_weights, topk_indices);
}

} // namespace infinilm::models::deepseek_v4::moe_backends
