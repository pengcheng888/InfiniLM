#include "linear.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>

namespace {

bool row_parallel_allreduce_profile_enabled() {
    static const bool enabled = [] {
        const char *env = std::getenv("INFINILM_ROW_PARALLEL_ALLREDUCE_PROFILE");
        if (env == nullptr || env[0] == '\0') {
            return false;
        }
        const std::string value(env);
        return value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON";
    }();
    return enabled;
}

size_t row_parallel_allreduce_profile_limit() {
    static const size_t limit = [] {
        const char *env = std::getenv("INFINILM_ROW_PARALLEL_ALLREDUCE_PROFILE_LIMIT");
        if (env == nullptr || env[0] == '\0') {
            return size_t{128};
        }
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end == env) {
            return size_t{128};
        }
        return static_cast<size_t>(parsed);
    }();
    return limit;
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string shape_to_string(const infinicore::Shape &shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace

namespace infinilm::nn {

// ---- Linear ----

Linear::Linear(size_t in_features, size_t out_features, bool bias,
               const infinicore::DataType &dtype, const infinicore::Device &device)
    : BaseLinear(in_features, out_features,
                 std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
                 bias, dtype, device, -1, 0, 1) {
}

Linear::Linear(size_t in_features, size_t out_features,
               std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
               bool bias, const infinicore::DataType &dtype, const infinicore::Device &device)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device, -1, 0, 1) {
}

infinicore::Tensor Linear::forward(infinicore::Tensor &input) const {
    return BaseLinear::forward(input);
}

void Linear::forward_(infinicore::Tensor output, infinicore::Tensor &input) const {
    BaseLinear::forward_(output, input);
}

std::string Linear::extra_repr() const {
    return "Linear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

// ---- ColumnParallelLinear ----

ColumnParallelLinear::ColumnParallelLinear(size_t in_features, size_t out_features, bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           infinicore::Size tp_rank, infinicore::Size tp_size,
                                           int tp_num_heads)
    : BaseLinear(in_features, out_features,
                 std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
                 bias, dtype, device, 0, tp_rank, tp_size, tp_num_heads),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {
}

ColumnParallelLinear::ColumnParallelLinear(size_t in_features, size_t out_features,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           bool bias, const infinicore::DataType &dtype, const infinicore::Device &device,
                                           infinicore::Size tp_rank, infinicore::Size tp_size,
                                           int tp_num_heads)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device,
                 0, tp_rank, tp_size, tp_num_heads),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {
}

infinicore::Tensor ColumnParallelLinear::forward(infinicore::Tensor &input) const {
    return BaseLinear::forward(input);
}

void ColumnParallelLinear::forward_(infinicore::Tensor output, infinicore::Tensor &input) const {
    BaseLinear::forward_(output, input);
}

std::string ColumnParallelLinear::extra_repr() const {
    return "ColumnParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

// ---- RowParallelLinear ----

RowParallelLinear::RowParallelLinear(size_t in_features, size_t out_features, bool bias,
                                     const infinicore::DataType &dtype, const infinicore::Device &device,
                                     infinicore::Size tp_rank, infinicore::Size tp_size,
                                     infinicclComm_t communicator)
    : BaseLinear(in_features, out_features,
                 std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
                 bias, dtype, device, 1, tp_rank, tp_size),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {
}

RowParallelLinear::RowParallelLinear(size_t in_features, size_t out_features,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     bool bias, const infinicore::DataType &dtype, const infinicore::Device &device,
                                     infinicore::Size tp_rank, infinicore::Size tp_size,
                                     infinicclComm_t communicator)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device,
                 1, tp_rank, tp_size),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {
}

infinicore::Tensor RowParallelLinear::forward(infinicore::Tensor &input) const {
    if (tp_size_ > 1 && communicator_ != nullptr) {
        return compute_linear_allreduce(input, communicator_);
    }
    return BaseLinear::forward(input);
}

void RowParallelLinear::forward_(infinicore::Tensor output, infinicore::Tensor &input) const {
    BaseLinear::forward_(output, input);

    if ((tp_size_ > 1) && (communicator_ != nullptr)) {
        if (row_parallel_allreduce_profile_enabled()) {
            static std::atomic<size_t> log_count{0};
            const size_t seq = log_count.fetch_add(1, std::memory_order_relaxed);
            const bool should_log = seq < row_parallel_allreduce_profile_limit();
            const auto t0 = std::chrono::steady_clock::now();
            infinicore::context::syncStream();
            const auto t1 = std::chrono::steady_clock::now();
            infinicore::op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator_);
            const auto t2 = std::chrono::steady_clock::now();
            infinicore::context::syncStream();
            const auto t3 = std::chrono::steady_clock::now();
            if (should_log) {
                spdlog::info(
                    "[INFINILM_ROW_PARALLEL_ALLREDUCE_PROFILE] seq={} tp_rank={} device={} shape={} numel={} bytes={} pre_sync_ms={:.6f} call_host_ms={:.6f} post_sync_ms={:.6f}",
                    seq,
                    tp_rank_,
                    output->device().getIndex(),
                    shape_to_string(output->shape()),
                    output->numel(),
                    output->nbytes(),
                    elapsed_ms(t0, t1),
                    elapsed_ms(t1, t2),
                    elapsed_ms(t2, t3));
            }
        } else {
            infinicore::op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator_);
        }
    }
}

std::string RowParallelLinear::extra_repr() const {
    return "RowParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinilm::nn
