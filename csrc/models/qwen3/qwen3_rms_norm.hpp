#pragma once

#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

namespace infinilm::models::qwen3 {

class Qwen3RMSNorm : public infinicore::nn::Module {
public:
    Qwen3RMSNorm(size_t normalized_shape,
                 double eps,
                 const infinicore::DataType &dtype,
                 const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &x) const;
    void forward_inplace(infinicore::Tensor &x, infinicore::Tensor &residual) const;

    infinicore::Tensor weight() const { return weight_; }
    float eps() const { return static_cast<float>(eps_); }

private:
    INFINICORE_NN_PARAMETER(weight);
    size_t normalized_shape_;
    double eps_;
    infinicore::DataType dtype_;
};

} // namespace infinilm::models::qwen3
