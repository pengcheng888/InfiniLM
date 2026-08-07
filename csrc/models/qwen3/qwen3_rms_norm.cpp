#include "qwen3_rms_norm.hpp"

#include "infinicore/ops/qwen3_add_rms_norm.hpp"
#include "infinicore/ops/qwen3_rms_norm.hpp"

namespace infinilm::models::qwen3 {

Qwen3RMSNorm::Qwen3RMSNorm(size_t normalized_shape,
                           double eps,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device)
    : normalized_shape_(normalized_shape), eps_(eps), dtype_(dtype) {
    device_ = device;
    INFINICORE_NN_PARAMETER_INIT(weight, ({normalized_shape_}, dtype_, device));
}

infinicore::Tensor Qwen3RMSNorm::forward(const infinicore::Tensor &x) const {
    return infinicore::op::qwen3_rms_norm(x, weight_, static_cast<float>(eps_));
}

void Qwen3RMSNorm::forward_inplace(infinicore::Tensor &x, infinicore::Tensor &residual) const {
    if (!residual) {
        residual = x;
        x = infinicore::op::qwen3_rms_norm(x, weight_, static_cast<float>(eps_));
        return;
    }
    infinicore::op::qwen3_add_rms_norm_inplace(x, residual, weight_, static_cast<float>(eps_));
}

} // namespace infinilm::models::qwen3
