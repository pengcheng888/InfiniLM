#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>
#include <utility>

namespace infinilm::layers::mla_attention::backends {

class FlashMLAImpl {
public:
    FlashMLAImpl(size_t num_heads,
                 size_t head_size,
                 float scale,
                 size_t num_kv_heads,
                 size_t layer_idx,
                 size_t head_dim_v);

    std::pair<infinicore::Tensor, infinicore::Tensor> forward_mqa(
        const infinicore::Tensor &query,
        const infinicore::Tensor &kv_c,
        const infinicore::Tensor &k_pe) const;

    void do_kv_cache_update(const infinicore::Tensor &kv_c,
                            const infinicore::Tensor &k_pe) const;

private:
    void do_kv_cache_update(const infinicore::Tensor &kv_c,
                            const infinicore::Tensor &k_pe,
                            infinicore::Tensor &kv_cache,
                            const infinicore::Tensor &slot_mapping) const;

    size_t num_heads_;
    size_t head_size_;
    float scale_;
    size_t num_kv_heads_;
    size_t layer_idx_;
    size_t head_dim_v_;
};

} // namespace infinilm::layers::mla_attention::backends
