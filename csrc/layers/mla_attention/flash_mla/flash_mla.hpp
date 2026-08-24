#pragma once

#include "../../../global_state/flash_mla_sched_meta.hpp"
#include "infinicore/tensor.hpp"

#include <cstdint>
#include <optional>
#include <utility>

namespace infinilm::layers::mla_attention {

std::pair<infinicore::Tensor, infinicore::Tensor> flash_mla_with_kvcache(
    const infinicore::Tensor &q,
    const infinicore::Tensor &k_cache,
    std::optional<infinicore::Tensor> block_table,
    std::optional<infinicore::Tensor> cache_seqlens,
    int64_t head_dim_v,
    infinilm::global_state::FlashMLASchedMeta &tile_scheduler_metadata,
    std::optional<infinicore::Tensor> num_splits,
    std::optional<double> softmax_scale,
    bool causal,
    bool is_fp8_kvcache,
    std::optional<infinicore::Tensor> indices,
    std::optional<infinicore::Tensor> attn_sink,
    std::optional<infinicore::Tensor> extra_k_cache,
    std::optional<infinicore::Tensor> extra_indices_in_kvcache,
    std::optional<infinicore::Tensor> topk_length,
    std::optional<infinicore::Tensor> extra_topk_length);

} // namespace infinilm::layers::mla_attention
