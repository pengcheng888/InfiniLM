#pragma once

#include "infinicore/ops/flash_mla/flash_mla_sched_meta/flash_mla_sched_meta.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <utility>

namespace infinilm::layers::mla_attention {

std::pair<infinicore::Tensor, infinicore::Tensor> compute_dense_flash_mla(
    const infinicore::Tensor &query,
    const infinicore::Tensor &kv_cache,
    const infinicore::Tensor &block_tables,
    const infinicore::Tensor &total_sequence_lengths,
    size_t head_dim_v,
    infinicore::op::flash_mla::FlashMLASchedMeta &sched_meta,
    float softmax_scale);

std::pair<infinicore::Tensor, infinicore::Tensor> compute_sparse_flash_mla(
    const infinicore::Tensor &query,
    const infinicore::Tensor &kv_cache,
    const infinicore::Tensor &indices,
    const infinicore::Tensor &attn_sink,
    const infinicore::Tensor &topk_lengths,
    size_t head_dim_v,
    infinicore::op::flash_mla::FlashMLASchedMeta &sched_meta,
    float softmax_scale);

} // namespace infinilm::layers::mla_attention
