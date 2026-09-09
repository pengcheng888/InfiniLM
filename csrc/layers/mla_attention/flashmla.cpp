#include "flashmla.hpp"

#include "infinicore/ops/flash_mla/flash_mla_with_kvcache.hpp"

#include <cstdint>
#include <optional>

namespace infinilm::layers::mla_attention {

std::pair<infinicore::Tensor, infinicore::Tensor> compute_dense_flash_mla(
    const infinicore::Tensor &query,
    const infinicore::Tensor &kv_cache,
    const infinicore::Tensor &block_tables,
    const infinicore::Tensor &total_sequence_lengths,
    size_t head_dim_v,
    infinicore::op::flash_mla::FlashMLASchedMeta &sched_meta,
    float softmax_scale) {
    return infinicore::op::flash_mla::flash_mla_with_kvcache(
        query,
        kv_cache,
        block_tables,
        total_sequence_lengths,
        static_cast<int64_t>(head_dim_v),
        sched_meta,
        std::nullopt,
        static_cast<double>(softmax_scale),
        false,
        false);
}

std::pair<infinicore::Tensor, infinicore::Tensor> compute_sparse_flash_mla(
    const infinicore::Tensor &query,
    const infinicore::Tensor &kv_cache,
    const infinicore::Tensor &indices,
    const infinicore::Tensor &attn_sink,
    const infinicore::Tensor &topk_lengths,
    size_t head_dim_v,
    infinicore::op::flash_mla::FlashMLASchedMeta &sched_meta,
    float softmax_scale) {
    return infinicore::op::flash_mla::flash_mla_with_kvcache(
        query,
        kv_cache,
        std::nullopt,
        std::nullopt,
        static_cast<int64_t>(head_dim_v),
        sched_meta,
        std::nullopt,
        static_cast<double>(softmax_scale),
        false,
        true,
        indices,
        attn_sink,
        std::nullopt,
        std::nullopt,
        topk_lengths,
        std::nullopt);
}

} // namespace infinilm::layers::mla_attention
