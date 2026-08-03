#pragma once

#include "../deepseek_v4_scratch.hpp"
#include "infinicore/tensor.hpp"

#include <optional>
#include <string>

namespace infinilm::models::deepseek_v4::moe_backends {

enum class RoutedExpertBackend {
    Naive,
    LmslimFused,
    FusedExpertsInt8Marlin,
    AiterSplit,
    LightopSplit,
};

struct RoutedExpertBackendChoice {
    RoutedExpertBackend backend{RoutedExpertBackend::FusedExpertsInt8Marlin};
    bool explicit_backend{false};
};

struct MarlinGemmConfig {
    int block_size{16};
    int gemm1_mode{54};
    int gemm2_mode{54};
    int delta{1};
    bool supported{false};
};

struct MarlinGemmOverride {
    int block_size{0};
    int mode{-1};
    int gemm1_mode{-1};
    int gemm2_mode{-1};
    int delta{-1};
};

struct RoutedExpertContext {
    size_t num_experts{0};
    size_t hidden_size{0};
    size_t intermediate_size_per_partition{0};
    size_t num_experts_per_tok{0};
    double routed_scaling_factor{1.0};
    int marlin_block_size{16};
    int marlin_mode{54};
    int marlin_delta{1};
    MarlinGemmOverride marlin_override;
    infinicore::Tensor w13_weight;
    infinicore::Tensor w13_weight_scale;
    infinicore::Tensor w2_weight;
    infinicore::Tensor w2_weight_scale;
    infinicore::Tensor w13_weight_marlin;
    infinicore::Tensor w2_weight_marlin;
    bool raw_weights_released{false};
};

RoutedExpertBackendChoice select_routed_expert_backend();
const char *to_string(RoutedExpertBackend backend);
bool requires_marlin_repack(RoutedExpertBackend backend);
MarlinGemmOverride read_marlin_gemm_override_from_env();
MarlinGemmConfig select_marlin_gemm_config(size_t num_tokens,
                                           size_t hidden_size,
                                           size_t intermediate_size,
                                           size_t top_k,
                                           int fallback_block_size,
                                           int fallback_mode,
                                           int fallback_delta,
                                           const MarlinGemmOverride &override_config);

infinicore::Tensor forward_routed_experts(RoutedExpertBackendChoice choice,
                                          const RoutedExpertContext &ctx,
                                          DeepseekV4RoutedExpertScratch &scratch,
                                          const infinicore::Tensor &hidden_states,
                                          const infinicore::Tensor &topk_weights,
                                          const infinicore::Tensor &topk_indices,
                                          const std::optional<infinicore::Tensor> &shared_output = std::nullopt);

} // namespace infinilm::models::deepseek_v4::moe_backends
