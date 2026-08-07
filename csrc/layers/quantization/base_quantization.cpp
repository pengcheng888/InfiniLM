#include "base_quantization.hpp"

#include "infinicore/ops/blas_copy.hpp"

namespace infinilm::quantization {

void BaseQuantization::forward_(
    const ParamsMap &params,
    infinicore::Tensor output,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha) const {
    auto computed = forward(params, input, has_bias, alpha);
    infinicore::op::blas_copy_(computed, output);
}

} // namespace infinilm::quantization
