#include "parallel_lm_head.hpp"

#include "infinicore/ops/distributed/vocab_parallel_logits_gather.hpp"

#include <stdexcept>

namespace infinilm::layers::lm_head {

ParallelLMHead::ParallelLMHead(size_t hidden_size,
                               size_t vocab_size,
                               bool bias,
                               const infinicore::DataType &dtype,
                               const infinicore::Device &device,
                               infinicore::Size tp_rank,
                               infinicore::Size tp_size,
                               infinicclComm_t communicator)
    : infinilm::layers::linear::ColumnParallelLinear(
          hidden_size,
          vocab_size,
          bias,
          dtype,
          device,
          tp_rank,
          tp_size),
      vocab_size_(vocab_size),
      tp_size_(tp_size),
      communicator_(communicator) {
    if (tp_size_ == 0) {
        throw std::runtime_error("ParallelLMHead: tp_size must be positive.");
    }
    if (vocab_size_ % tp_size_ != 0) {
        throw std::runtime_error("ParallelLMHead: vocab_size must be divisible by tp_size in the current implementation.");
    }
}

infinicore::Tensor ParallelLMHead::forward_local(infinicore::Tensor &input) const {
    return infinilm::layers::linear::ColumnParallelLinear::forward(input);
}

infinicore::Tensor ParallelLMHead::ensure_gather_workspace(const infinicore::Tensor &local_logits) const {
    const auto required = infinicore::op::distributed::vocab_parallel_logits_gather_workspace_numel(
        local_logits, tp_size_);
    if (required == 0) {
        return infinicore::Tensor();
    }
    if (!logits_gather_workspace_
        || logits_gather_workspace_->numel() < required
        || logits_gather_workspace_->dtype() != local_logits->dtype()
        || logits_gather_workspace_->device().getType() != local_logits->device().getType()
        || logits_gather_workspace_->device().getIndex() != local_logits->device().getIndex()) {
        logits_gather_workspace_ = infinicore::Tensor::empty(
            {required}, local_logits->dtype(), local_logits->device());
    }
    return logits_gather_workspace_;
}

infinicore::Tensor ParallelLMHead::gather_logits(const infinicore::Tensor &local_logits) const {
    if (tp_size_ <= 1 || communicator_ == nullptr) {
        return local_logits;
    }
    if (!local_logits || local_logits->ndim() != 2) {
        throw std::runtime_error("ParallelLMHead::gather_logits expects local logits [tokens, vocab_per_rank].");
    }

    auto shape = local_logits->shape();
    shape[1] *= tp_size_;
    auto logits = infinicore::Tensor::empty(shape, local_logits->dtype(), local_logits->device());
    auto workspace = ensure_gather_workspace(local_logits);
    infinicore::op::distributed::vocab_parallel_logits_gather_(
        logits, local_logits, workspace, communicator_);
    if (logits->size(1) != vocab_size_) {
        throw std::runtime_error("ParallelLMHead::gather_logits produced unexpected vocab dimension.");
    }
    return logits;
}

infinicore::Tensor ParallelLMHead::forward(infinicore::Tensor &input) const {
    auto local_logits = forward_local(input);
    return gather_logits(local_logits);
}

} // namespace infinilm::layers::lm_head
