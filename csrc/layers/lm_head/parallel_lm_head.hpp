#pragma once

#include "../linear/linear.hpp"

#include <infiniccl.h>

namespace infinilm::layers::lm_head {

class ParallelLMHead : public infinilm::layers::linear::ColumnParallelLinear {
public:
    ParallelLMHead(size_t hidden_size,
                   size_t vocab_size,
                   bool bias,
                   const infinicore::DataType &dtype,
                   const infinicore::Device &device,
                   infinicore::Size tp_rank = 0,
                   infinicore::Size tp_size = 1,
                   infinicclComm_t communicator = nullptr);

    infinicore::Tensor forward(infinicore::Tensor &input) const;
    infinicore::Tensor forward_local(infinicore::Tensor &input) const;
    infinicore::Tensor gather_logits(const infinicore::Tensor &local_logits) const;

private:
    infinicore::Tensor ensure_gather_workspace(const infinicore::Tensor &local_logits) const;

    size_t vocab_size_{0};
    infinicore::Size tp_size_{1};
    infinicclComm_t communicator_{nullptr};
    mutable infinicore::Tensor logits_gather_workspace_;
};

} // namespace infinilm::layers::lm_head
