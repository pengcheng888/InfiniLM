#include "fused_linear.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/deepseek_v4_lightop_linear_w8a8_asm.hpp"
#include "infinicore/ops/deepseek_v4_lightop_linear_w8a8_smooth.hpp"
#include "infinicore/ops/deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8.hpp"
#include "infinicore/ops/deepseek_v4_lmslim_linear_w8a8.hpp"
#include "infinicore/ops/deepseek_v4_lmslim_rocblas_linear_w8a8.hpp"
#include "infinicore/ops/linear_w8a8i8.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::layers::linear {
namespace {

bool env_flag(const char *name, bool default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' || value[0] == 'n' || value[0] == 'N');
}

enum class W8A8WorkspaceBackend {
    Off,
    Native,
    Lmslim,
    LmslimRocblas,
    LmslimHipblaslt,
    LightopSmooth,
};

std::ostream &operator<<(std::ostream &os, W8A8WorkspaceBackend backend) {
    switch (backend) {
    case W8A8WorkspaceBackend::Off:
        return os << "off";
    case W8A8WorkspaceBackend::Native:
        return os << "native";
    case W8A8WorkspaceBackend::Lmslim:
        return os << "lmslim";
    case W8A8WorkspaceBackend::LmslimRocblas:
        return os << "lmslim_rocblas";
    case W8A8WorkspaceBackend::LmslimHipblaslt:
        return os << "lmslim_hipblaslt";
    case W8A8WorkspaceBackend::LightopSmooth:
        return os << "lightop_smooth";
    }
    return os << "unknown";
}

W8A8WorkspaceBackend w8a8_workspace_backend() {
    static const W8A8WorkspaceBackend backend = [] {
        const char *value = std::getenv("INFINILM_FUSED_REPLICATED_LINEAR_W8A8_BACKEND");
        if (value == nullptr || value[0] == '\0') {
            return env_flag("INFINILM_FUSED_REPLICATED_LINEAR_LMSLIM", true)
                     ? W8A8WorkspaceBackend::Native
                     : W8A8WorkspaceBackend::Off;
        }
        const std::string name(value);
        if (name == "off" || name == "none" || name == "0" || name == "false" || name == "FALSE") {
            return W8A8WorkspaceBackend::Off;
        }
        if (name == "native" || name == "linear_w8a8i8") {
            return W8A8WorkspaceBackend::Native;
        }
        if (name == "lmslim") {
            return W8A8WorkspaceBackend::Lmslim;
        }
        if (name == "lmslim_rocblas" || name == "rocblas") {
            return W8A8WorkspaceBackend::LmslimRocblas;
        }
        if (name == "lmslim_hipblaslt" || name == "hipblaslt" || name == "hipblaslt_channelwise") {
            return W8A8WorkspaceBackend::LmslimHipblaslt;
        }
        if (name == "lightop_smooth" || name == "lmslim_lightop") {
            if (!env_flag("INFINILM_FUSED_REPLICATED_LINEAR_ALLOW_EXPERIMENTAL_LIGHTOP_SMOOTH", false)) {
                throw std::runtime_error(
                    "INFINILM_FUSED_REPLICATED_LINEAR_W8A8_BACKEND=lightop_smooth is currently an experimental "
                    "full lightop smooth linear path. Single-op eager/graph tests pass, but InfiniLM TP8 "
                    "end-to-end still crashes after the lightop SO GEMM path. Set "
                    "INFINILM_FUSED_REPLICATED_LINEAR_ALLOW_EXPERIMENTAL_LIGHTOP_SMOOTH=1 only for isolated "
                    "debugging, or use native/lmslim for runnable inference.");
            }
            return W8A8WorkspaceBackend::LightopSmooth;
        }
        if (name == "lightop_asm" || name == "lightop") {
            throw std::runtime_error(
                "INFINILM_FUSED_REPLICATED_LINEAR_W8A8_BACKEND=lightop_asm is disabled for InfiniLM "
                "because the current lightop SO GEMM bridge segfaults in end-to-end forward. "
                "Use native, lmslim, or lightop_smooth.");
        }
        throw std::runtime_error(
            "Unsupported INFINILM_FUSED_REPLICATED_LINEAR_W8A8_BACKEND: " + name + ". Supported values: native, lmslim, lmslim_rocblas, lmslim_hipblaslt, lightop_smooth, off.");
    }();
    return backend;
}

bool w8a8_workspace_linear_enabled() {
    static const bool enabled = w8a8_workspace_backend() != W8A8WorkspaceBackend::Off;
    return enabled;
}

const char *w8a8_workspace_backend_name() {
    switch (w8a8_workspace_backend()) {
    case W8A8WorkspaceBackend::Off:
        return "off";
    case W8A8WorkspaceBackend::Native:
        return "native";
    case W8A8WorkspaceBackend::Lmslim:
        return "lmslim";
    case W8A8WorkspaceBackend::LmslimRocblas:
        return "lmslim_rocblas";
    case W8A8WorkspaceBackend::LmslimHipblaslt:
        return "lmslim_hipblaslt";
    case W8A8WorkspaceBackend::LightopSmooth:
        return "lightop_smooth";
    }
    return "unknown";
}

bool w8a8_workspace_linear_profile_enabled() {
    static const bool enabled = env_flag("INFINILM_FUSED_REPLICATED_LINEAR_PROFILE", false);
    return enabled;
}

bool w8a8_workspace_linear_timing_enabled() {
    static const bool enabled = env_flag("INFINILM_FUSED_REPLICATED_LINEAR_TIMING", false);
    return enabled;
}

struct W8A8WorkspaceLinearStats {
    std::atomic<unsigned long long> hits{0};
    std::atomic<unsigned long long> misses{0};
    std::atomic<unsigned long long> hit_tokens{0};
    std::atomic<unsigned long long> timed_calls{0};
    std::atomic<unsigned long long> total_us{0};
};

W8A8WorkspaceLinearStats &w8a8_workspace_linear_stats() {
    static W8A8WorkspaceLinearStats stats;
    return stats;
}

void dump_w8a8_workspace_linear_stats() {
    const auto &stats = w8a8_workspace_linear_stats();
    const auto hits = stats.hits.load(std::memory_order_relaxed);
    const auto misses = stats.misses.load(std::memory_order_relaxed);
    const auto tokens = stats.hit_tokens.load(std::memory_order_relaxed);
    const auto timed_calls = stats.timed_calls.load(std::memory_order_relaxed);
    const auto total_us = stats.total_us.load(std::memory_order_relaxed);
    const double total_ms = static_cast<double>(total_us) / 1000.0;
    const double avg_ms = timed_calls == 0 ? 0.0 : total_ms / static_cast<double>(timed_calls);
    std::fprintf(stderr,
                 "[INFINILM_FUSED_REPLICATED_LINEAR_PROFILE] w8a8_workspace backend=%s hits=%llu misses=%llu hit_tokens=%llu timed_calls=%llu total_ms=%.3f avg_ms=%.6f\n",
                 w8a8_workspace_backend_name(),
                 hits,
                 misses,
                 tokens,
                 timed_calls,
                 total_ms,
                 avg_ms);
}

void ensure_w8a8_workspace_linear_stats_registered() {
    static std::once_flag flag;
    std::call_once(flag, [] { std::atexit(dump_w8a8_workspace_linear_stats); });
}

} // namespace

// ---------------------------------------------------------
// Fused Replicated Linear
// ---------------------------------------------------------
FusedReplicatedLinear::FusedReplicatedLinear(size_t hidden_size,
                                             size_t split_size,
                                             const std::string &first_name,
                                             const std::string &second_name,
                                             RegisterParamFn register_fn,
                                             const infinicore::DataType &dtype,
                                             const infinicore::Device &device)
    : FusedReplicatedLinear(hidden_size,
                            split_size,
                            split_size,
                            first_name,
                            second_name,
                            std::move(register_fn),
                            std::make_shared<infinilm::quantization::NoneQuantization>(),
                            false,
                            dtype,
                            device) {
}

FusedReplicatedLinear::FusedReplicatedLinear(size_t hidden_size,
                                             size_t first_size,
                                             size_t second_size,
                                             const std::string &first_name,
                                             const std::string &second_name,
                                             RegisterParamFn register_fn,
                                             std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                             bool bias,
                                             const infinicore::DataType &dtype,
                                             const infinicore::Device &device)
    : infinilm::nn::Linear(
          hidden_size,
          first_size + second_size,
          quantization ? std::move(quantization) : std::make_shared<infinilm::quantization::NoneQuantization>(),
          bias,
          dtype,
          device),
      first_size_(first_size),
      second_size_(second_size),
      first_name_(first_name),
      second_name_(second_name),
      register_fn_(std::move(register_fn)) {
    if (!register_fn_) {
        throw std::runtime_error("FusedReplicatedLinear requires a parameter register function");
    }
    register_split_params();
}

void FusedReplicatedLinear::register_split_params() {
    if (!register_fn_) {
        throw std::runtime_error("FusedReplicatedLinear requires a parameter register function");
    }
    const std::string key_name = parameters_.count("qweight") ? "qweight" : "weight";
    const auto &key_param = get_parameter_ref(key_name);
    const int fused_dim = get_quantization()->get_fused_split_dim();
    const size_t logical_output = get_quantization()->get_logical_dim_size(key_param->size(static_cast<size_t>(fused_dim)));
    const size_t expected_output = first_size_ + second_size_;
    if (logical_output != expected_output) {
        throw std::runtime_error("FusedReplicatedLinear split sizes do not match fused output size");
    }

    split_infos_ = {
        {first_name_, 0, first_size_},
        {second_name_, first_size_, second_size_},
    };
    auto params = get_quantization()->split_params(parameters_, split_infos_, fused_dim, 0, 1, -1);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }
}

void FusedReplicatedLinear::process_weights_after_loading() {
    infinilm::nn::BaseLinear::process_weights_after_loading();
    register_split_params();
}

infinicore::Tensor FusedReplicatedLinear::forward(const infinicore::Tensor &input) const {
    if (input->ndim() != 2 || input->size(1) != in_features()) {
        throw std::runtime_error("FusedReplicatedLinear::forward expects input [tokens, hidden_size]");
    }
    auto input_ref = input;
    return infinilm::nn::Linear::forward(input_ref);
}

void FusedReplicatedLinear::forward_(infinicore::Tensor output, const infinicore::Tensor &input) const {
    if (input->ndim() != 2 || input->size(1) != in_features()) {
        throw std::runtime_error("FusedReplicatedLinear::forward_ expects input [tokens, hidden_size]");
    }
    if (output->ndim() != 2 || output->size(0) != input->size(0) || output->size(1) != out_features()) {
        throw std::runtime_error("FusedReplicatedLinear::forward_ output shape mismatch");
    }
    auto input_ref = input;
    infinilm::nn::Linear::forward_(output, input_ref);
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
FusedReplicatedLinear::forward_split(const infinicore::Tensor &input) const {
    const bool use_w8a8_workspace_linear = w8a8_workspace_linear_enabled();
    const bool profile_w8a8_workspace_linear = w8a8_workspace_linear_profile_enabled();
    if (profile_w8a8_workspace_linear) {
        ensure_w8a8_workspace_linear_stats_registered();
    }
    const bool can_use_w8a8_workspace_linear = get_quantization()->get_quant_scheme() == infinilm::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8 && input->ndim() == 2 && input->size(0) > 16 && input->size(1) == in_features() && input->is_contiguous() && in_features() == 4096 && out_features() == 1536 && input->dtype() == infinicore::DataType::BF16 && !has_bias() && std::fabs(alpha() - 1.0f) <= 1e-7f;
    if (use_w8a8_workspace_linear && can_use_w8a8_workspace_linear) {
        auto input_contiguous = input;
        const size_t tokens = input_contiguous->size(0);
        const infinicore::Shape output_shape = {tokens, out_features()};
        const infinicore::Shape q_input_shape = {tokens, in_features()};
        const infinicore::Shape input_scale_shape = {tokens, 1};
        const infinicore::Shape smooth_scale_shape = {in_features()};

        if (!w8a8_output_workspace_ || w8a8_output_workspace_->shape() != output_shape || w8a8_output_workspace_->dtype() != input_contiguous->dtype() || w8a8_output_workspace_->device() != input_contiguous->device()) {
            w8a8_output_workspace_ = infinicore::Tensor::empty(
                output_shape, input_contiguous->dtype(), input_contiguous->device());
        }
        if (!w8a8_q_input_workspace_ || w8a8_q_input_workspace_->shape() != q_input_shape || w8a8_q_input_workspace_->dtype() != infinicore::DataType::I8 || w8a8_q_input_workspace_->device() != input_contiguous->device()) {
            w8a8_q_input_workspace_ = infinicore::Tensor::empty(
                q_input_shape, infinicore::DataType::I8, input_contiguous->device());
        }
        if (!w8a8_input_scale_workspace_ || w8a8_input_scale_workspace_->shape() != input_scale_shape || w8a8_input_scale_workspace_->dtype() != infinicore::DataType::F32 || w8a8_input_scale_workspace_->device() != input_contiguous->device()) {
            w8a8_input_scale_workspace_ = infinicore::Tensor::empty(
                input_scale_shape, infinicore::DataType::F32, input_contiguous->device());
        }
        if (w8a8_workspace_backend() == W8A8WorkspaceBackend::LmslimRocblas && (!w8a8_accum_workspace_ || w8a8_accum_workspace_->shape() != output_shape || w8a8_accum_workspace_->dtype() != infinicore::DataType::I32 || w8a8_accum_workspace_->device() != input_contiguous->device())) {
            w8a8_accum_workspace_ = infinicore::Tensor::empty(
                output_shape, infinicore::DataType::I32, input_contiguous->device());
        }
        if ((w8a8_workspace_backend() == W8A8WorkspaceBackend::Lmslim || w8a8_workspace_backend() == W8A8WorkspaceBackend::LmslimRocblas || w8a8_workspace_backend() == W8A8WorkspaceBackend::LmslimHipblaslt || w8a8_workspace_backend() == W8A8WorkspaceBackend::LightopSmooth) && (!w8a8_smooth_scale_workspace_ || w8a8_smooth_scale_workspace_->shape() != smooth_scale_shape || w8a8_smooth_scale_workspace_->dtype() != infinicore::DataType::F32 || w8a8_smooth_scale_workspace_->device() != input_contiguous->device())) {
            w8a8_smooth_scale_workspace_ = infinicore::Tensor::ones(
                smooth_scale_shape, infinicore::DataType::F32, input_contiguous->device());
        }

        if (profile_w8a8_workspace_linear) {
            auto &stats = w8a8_workspace_linear_stats();
            stats.hits.fetch_add(1, std::memory_order_relaxed);
            stats.hit_tokens.fetch_add(tokens, std::memory_order_relaxed);
        }
        auto run_w8a8_workspace_linear = [&]() {
            auto back = w8a8_workspace_backend();
            std::cout << "-----> back: " << back << std::endl;
            switch (back) {
            case W8A8WorkspaceBackend::Native:
                infinicore::op::linear_w8a8i8_out_workspace_(
                    w8a8_output_workspace_,
                    input_contiguous,
                    weight(),
                    weight_scale(),
                    std::nullopt,
                    w8a8_q_input_workspace_,
                    w8a8_input_scale_workspace_);
                return;
            case W8A8WorkspaceBackend::Lmslim:
                infinicore::op::deepseek_v4_lmslim_linear_w8a8_(
                    w8a8_output_workspace_,
                    input_contiguous,
                    weight()->permute({1, 0}),
                    weight_scale(),
                    std::nullopt,
                    w8a8_q_input_workspace_,
                    w8a8_input_scale_workspace_,
                    w8a8_smooth_scale_workspace_);
                return;
            case W8A8WorkspaceBackend::LmslimRocblas:
                infinicore::op::deepseek_v4_lmslim_rocblas_linear_w8a8_(
                    w8a8_output_workspace_,
                    input_contiguous,
                    weight()->permute({1, 0}),
                    weight_scale(),
                    std::nullopt,
                    w8a8_q_input_workspace_,
                    w8a8_input_scale_workspace_,
                    w8a8_accum_workspace_,
                    w8a8_smooth_scale_workspace_);
                return;
            case W8A8WorkspaceBackend::LmslimHipblaslt:
                infinicore::op::deepseek_v4_lmslim_hipblaslt_channelwise_linear_w8a8_(
                    w8a8_output_workspace_,
                    input_contiguous,
                    weight(),
                    weight_scale(),
                    std::nullopt,
                    w8a8_q_input_workspace_,
                    w8a8_input_scale_workspace_,
                    w8a8_smooth_scale_workspace_);
                return;
            case W8A8WorkspaceBackend::LightopSmooth:
                infinicore::op::deepseek_v4_lightop_linear_w8a8_smooth_(
                    w8a8_output_workspace_,
                    input_contiguous,
                    weight(),
                    weight_scale(),
                    std::nullopt,
                    w8a8_q_input_workspace_,
                    w8a8_input_scale_workspace_,
                    w8a8_smooth_scale_workspace_);
                return;
            case W8A8WorkspaceBackend::Off:
                break;
            }
            throw std::runtime_error("W8A8 workspace linear backend is disabled.");
        };

        if (w8a8_workspace_linear_timing_enabled()) {
            infinicore::context::syncStream();
            const auto start = std::chrono::steady_clock::now();
            run_w8a8_workspace_linear();
            infinicore::context::syncStream();
            const auto end = std::chrono::steady_clock::now();
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            auto &stats = w8a8_workspace_linear_stats();
            stats.timed_calls.fetch_add(1, std::memory_order_relaxed);
            stats.total_us.fetch_add(static_cast<unsigned long long>(us), std::memory_order_relaxed);
        } else {
            run_w8a8_workspace_linear();
        }

        auto flat = w8a8_output_workspace_->view({tokens * out_features()});
        auto first = flat->narrow({{0, 0, tokens * first_size_}})->view({tokens, first_size_});
        auto second = flat->narrow({{0, tokens * first_size_, tokens * second_size_}})->view({tokens, second_size_});
        return {first, second};
    }
    if (profile_w8a8_workspace_linear) {
        w8a8_workspace_linear_stats().misses.fetch_add(1, std::memory_order_relaxed);
    }

    auto output = forward(input);
    auto first = output->narrow({{1, 0, first_size_}});
    auto second = output->narrow({{1, first_size_, second_size_}});
    return {first, second};
}

// ---------------------------------------------------------
// QKV Parallel Linear
// ---------------------------------------------------------
QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t head_dim,
                                     size_t num_q_head,
                                     size_t num_kv_head,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     bool bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(hidden_size,
                        head_dim, head_dim, head_dim,
                        num_q_head, num_kv_head, num_kv_head,
                        bias, bias, bias,
                        quantization,
                        dtype, device, rank_info) {}

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t q_dim, size_t k_dim, size_t v_dim,
                                     size_t num_q_head, size_t num_k_head, size_t num_v_head,
                                     bool q_bias, bool k_bias, bool v_bias,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : infinilm::nn::ColumnParallelLinear(
          hidden_size,
          calculate_out_feature_size(num_q_head, q_dim, num_k_head, k_dim, num_v_head, v_dim, rank_info),
          quantization == nullptr ? std::make_shared<infinilm::quantization::NoneQuantization>() : quantization,
          (q_bias || k_bias || v_bias),
          dtype,
          device,
          rank_info.tp_rank,
          rank_info.tp_size),
      q_dim_(q_dim),
      k_dim_(k_dim),
      v_dim_(v_dim),
      num_q_head_(num_q_head),
      num_k_head_(num_k_head),
      num_v_head_(num_v_head),
      q_bias_(q_bias),
      k_bias_(k_bias),
      v_bias_(v_bias),
      num_kv_head_replicas_(calculate_kv_replicas(num_k_head, rank_info.tp_size)) {

    if ((q_bias_ != k_bias_) || (k_bias_ != v_bias_)) {
        throw std::runtime_error("q_bias, k_bias, v_bias must all match");
    }

    q_out_size_ = num_q_head_ * q_dim_ / tp_size_;
    k_out_size_ = num_kv_head_replicas_ * num_k_head_ * k_dim_ / tp_size_;
    v_out_size_ = num_kv_head_replicas_ * num_v_head_ * v_dim_ / tp_size_;
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
QKVParallelLinear::forward_split(infinicore::Tensor &input) {
    auto output = this->forward(input);

    auto q_out = output->narrow({{2, 0, q_out_size_}});
    auto k_out = output->narrow({{2, q_out_size_, k_out_size_}});
    auto v_out = output->narrow({{2, q_out_size_ + k_out_size_, v_out_size_}});

    return std::make_tuple(q_out, k_out, v_out);
}

bool QKVParallelLinear::has_q_bias() const { return q_bias_; }
bool QKVParallelLinear::has_k_bias() const { return k_bias_; }
bool QKVParallelLinear::has_v_bias() const { return v_bias_; }

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t head_dim,
                                     size_t num_q_head, size_t num_kv_head,
                                     const std::string &q_name, const std::string &k_name, const std::string &v_name,
                                     RegisterParamFn register_fn,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     bool bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(hidden_size, head_dim, head_dim, head_dim, num_q_head, num_kv_head, num_kv_head, bias, bias, bias, q_name, k_name, v_name, register_fn, quantization, dtype, device, rank_info) {
}

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t q_dim, size_t k_dim, size_t v_dim,
                                     size_t num_q_head, size_t num_k_head, size_t num_v_head,
                                     bool q_bias, bool k_bias, bool v_bias,
                                     const std::string &q_name, const std::string &k_name, const std::string &v_name,
                                     RegisterParamFn register_fn,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(hidden_size, q_dim, k_dim, v_dim, num_q_head, num_k_head, num_v_head, q_bias, k_bias, v_bias, quantization, dtype, device, rank_info) {
    register_fn_ = register_fn;
    split_infos_ = {
        {q_name, 0, q_out_size_, 0},
        {k_name, q_out_size_, k_out_size_, num_k_head_},
        {v_name, q_out_size_ + k_out_size_, v_out_size_, num_v_head_},
    };
    auto params = this->split_params(split_infos_, tp_rank_, tp_size_, num_k_head_);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }
}

void QKVParallelLinear::process_weights_after_loading() {
    BaseLinear::process_weights_after_loading();
    if (register_fn_ && !split_infos_.empty()) {
        auto params = this->split_params(split_infos_, tp_rank_, tp_size_, num_k_head_);
        for (auto &sp : params) {
            register_fn_(sp.full_name, std::move(sp.param));
        }
    }
}

// ---------------------------------------------------------
// Gate-Up Parallel Linear
// ---------------------------------------------------------
GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, std::shared_ptr<infinilm::quantization::BaseQuantization> quantization, bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info)
    : GateUpParallelLinear(hidden_size, intermediate_size, bias, bias, quantization, dtype, device, rank_info) {
}

GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info)
    : infinilm::nn::ColumnParallelLinear(
          hidden_size,
          intermediate_size * 2,
          quantization == nullptr ? std::make_shared<infinilm::quantization::NoneQuantization>() : quantization,
          gate_bias || up_bias,
          dtype,
          device,
          rank_info.tp_rank,
          rank_info.tp_size),
      gate_bias_(gate_bias),
      up_bias_(up_bias) {
    if (gate_bias_ != up_bias_) {
        throw std::runtime_error("Not supported yet: gate_bias and up_bias should be given at the same time");
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> GateUpParallelLinear::forward_split(infinicore::Tensor &input) {
    auto output = this->forward(input);
    auto cols = output->shape()[2];
    auto gate_output = output->narrow({{2, 0, cols / 2}});
    auto up_output = output->narrow({{2, cols / 2, cols / 2}});
    return std::make_tuple(gate_output, up_output);
}

bool GateUpParallelLinear::has_gate_bias() const { return gate_bias_; }
bool GateUpParallelLinear::has_up_bias() const { return up_bias_; }

GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size,
                                           const std::string &gate_name, const std::string &up_name,
                                           RegisterParamFn register_fn,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info)
    : GateUpParallelLinear(hidden_size, intermediate_size, quantization, bias, dtype, device, rank_info) {
    const std::string &key_name = parameters_.count("qweight") ? "qweight" : "weight";
    const auto &key_param = get_parameter_ref(key_name);
    int fused_dim = this->get_quantization()->get_fused_split_dim();
    size_t logical_output = this->get_quantization()->get_logical_dim_size(key_param->size(fused_dim));
    size_t half_size = logical_output / 2;
    register_fn_ = register_fn;
    split_infos_ = {
        {gate_name, 0, half_size},
        {up_name, half_size, half_size},
    };
    auto params = this->split_params(split_infos_, tp_rank_, tp_size_, -1);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }
}

void GateUpParallelLinear::process_weights_after_loading() {
    BaseLinear::process_weights_after_loading();
    if (register_fn_ && !split_infos_.empty()) {
        auto params = this->split_params(split_infos_, tp_rank_, tp_size_, -1);
        for (auto &sp : params) {
            register_fn_(sp.full_name, std::move(sp.param));
        }
    }
}

} // namespace infinilm::layers::linear
