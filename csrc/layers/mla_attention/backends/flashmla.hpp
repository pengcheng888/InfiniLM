#pragma once

#include "infinicore/ops/flash_mla/flash_mla_sched_meta/flash_mla_sched_meta.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <utility>

namespace infinilm::layers::mla_attention::backends {

class DenseFlashMLAImpl {
public:
    DenseFlashMLAImpl(size_t num_heads,
                      size_t head_size,
                      float scale,
                      size_t num_kv_heads,
                      size_t layer_idx,
                      size_t head_dim_v);

    std::pair<infinicore::Tensor, infinicore::Tensor> forward_mqa(
        const infinicore::Tensor &query) const;

private:
    size_t num_heads_;
    size_t head_size_;
    float scale_;
    size_t num_kv_heads_;
    size_t layer_idx_;
    size_t head_dim_v_;
};

class SparseFlashMLAImpl {
public:
    SparseFlashMLAImpl(size_t num_heads,
                       size_t head_size,
                       float scale,
                       size_t num_kv_heads,
                       size_t head_dim_v);

    std::pair<infinicore::Tensor, infinicore::Tensor> forward_mqa(
        const infinicore::Tensor &query,
        const infinicore::Tensor &kv_cache,
        const infinicore::Tensor &indices,
        const infinicore::Tensor &attn_sink,
        const infinicore::Tensor &topk_lengths,
        infinicore::op::flash_mla::FlashMLASchedMeta &sched_meta) const;

private:
    size_t num_heads_;
    size_t head_size_;
    float scale_;
    size_t num_kv_heads_;
    size_t head_dim_v_;
};

} // namespace infinilm::layers::mla_attention::backends
