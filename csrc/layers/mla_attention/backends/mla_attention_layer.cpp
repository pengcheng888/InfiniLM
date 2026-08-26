#include "mla_attention_layer.hpp"

namespace infinilm::layers::mla_attention {

MLAAttentionLayer::MLAAttentionLayer(size_t num_heads,
                                     size_t head_size,
                                     float scale,
                                     size_t num_kv_heads,
                                     size_t layer_idx,
                                     size_t head_dim_v,
                                     infinicore::Tensor k_scale,
                                     infinicore::Tensor v_scale,
                                     ::infinilm::backends::AttentionBackend attn_backend) : k_scale_(k_scale), v_scale_(v_scale), layer_idx_(layer_idx) {
    switch (attn_backend) {
    case ::infinilm::backends::AttentionBackend::FLASHMLA:
        impl_ = std::make_shared<backends::FlashMLAImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx, head_dim_v);
        break;
    default:
        throw std::runtime_error("infinilm::layers::mla_attention::MLAAttentionLayer: unsupported attention backend");
    }
}

std::pair<infinicore::Tensor, infinicore::Tensor> MLAAttentionLayer::forward_mqa(
    const infinicore::Tensor &query,
    const infinicore::Tensor &kv_c,
    const infinicore::Tensor &k_pe) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.flashmla_attn_metadata;
    if (forward_context.kv_cache_vec.size() <= layer_idx_ || !forward_context.kv_cache_vec[layer_idx_]) {
        throw std::runtime_error("MLAAttentionLayer::forward_mqa requires MLA KV cache allocation");
    }
    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];

    return std::visit(
        [&](auto &impl_ptr) -> std::pair<infinicore::Tensor, infinicore::Tensor> {
            return impl_ptr->forward_mqa(*this, query, kv_c, k_pe, kv_cache, attn_metadata);
        },
        impl_);
}

void MLAAttentionLayer::do_kv_cache_update(const infinicore::Tensor &kv_c,
                                           const infinicore::Tensor &k_pe) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.flashmla_attn_metadata;
    if (forward_context.kv_cache_vec.size() <= layer_idx_ || !forward_context.kv_cache_vec[layer_idx_]) {
        throw std::runtime_error("MLAAttentionLayer::do_kv_cache_update requires MLA KV cache allocation");
    }
    if (!attn_metadata.slot_mapping) {
        throw std::runtime_error("MLAAttentionLayer::do_kv_cache_update requires slot_mapping");
    }
    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];

    std::visit(
        [&](auto &impl_ptr) {
            impl_ptr->do_kv_cache_update(kv_c, k_pe, kv_cache, attn_metadata.slot_mapping);
        },
        impl_);
}

} // namespace infinilm::layers::mla_attention
