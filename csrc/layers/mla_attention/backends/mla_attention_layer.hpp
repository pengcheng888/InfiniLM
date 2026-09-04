#pragma once

#include "../../../backends/attention_backends.hpp"
#include "../../../global_state/global_state.hpp"
#include "flashmla_v2.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <variant>

namespace infinilm::layers::mla_attention {
using MLAAttentionImpl = std::variant<std::shared_ptr<backends::FlashMLAV2Impl>>;

/**
 * @brief Attention layer.
 * This class takes query, key, and value tensors as input.
 * The input tensors can either contain prompt tokens or generation tokens.
 *
 * The class does the following:
 * - Update the KV cache.
 * - Perform (multi-head/multi-query/grouped-query) attention.
 * - Return the output tensor.
 */
class MLAAttentionLayer {
public:
    MLAAttentionLayer(size_t num_heads,
                      size_t head_size,
                      float scale,
                      size_t num_kv_heads,
                      size_t layer_idx,
                      size_t head_dim_v,
                      infinicore::Tensor k_scale,
                      infinicore::Tensor v_scale,
                      ::infinilm::backends::AttentionBackend attention_backend);

    std::pair<infinicore::Tensor, infinicore::Tensor> forward_mqa(
        const infinicore::Tensor &query,
        const infinicore::Tensor &kv_c,
        const infinicore::Tensor &k_pe) const;

    void do_kv_cache_update(const infinicore::Tensor &kv_c,
                            const infinicore::Tensor &k_pe) const;

    inline infinicore::Tensor get_k_scale() const { return k_scale_; }
    inline infinicore::Tensor get_v_scale() const { return v_scale_; }

private:
    infinicore::Tensor k_scale_;
    infinicore::Tensor v_scale_;
    size_t layer_idx_;
    MLAAttentionImpl impl_;
};
} // namespace infinilm::layers::mla_attention
