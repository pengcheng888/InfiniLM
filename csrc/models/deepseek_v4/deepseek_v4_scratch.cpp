#include "deepseek_v4_scratch.hpp"

#include <cassert>

namespace infinilm::models::deepseek_v4 {
namespace {

bool can_use_scratch_tensor(const infinicore::Tensor &scratch,
                            const infinicore::Shape &shape,
                            infinicore::DataType dtype,
                            const infinicore::Device &device) {
    if (!scratch || shape.empty()) {
        return false;
    }
    if (scratch->dtype() != dtype || scratch->device() != device) {
        return false;
    }
    const auto scratch_shape = scratch->shape();
    if (scratch_shape.size() != shape.size() || scratch_shape[0] < shape[0]) {
        return false;
    }
    for (size_t i = 1; i < shape.size(); ++i) {
        if (scratch_shape[i] != shape[i]) {
            return false;
        }
    }
    return true;
}

infinicore::Tensor get_scratch_or_empty(const infinicore::Tensor &scratch,
                                        const infinicore::Shape &shape,
                                        infinicore::DataType dtype,
                                        const infinicore::Device &device) {
    if (can_use_scratch_tensor(scratch, shape, dtype, device)) {
        return scratch->narrow({{0, 0, shape[0]}});
    }
    return infinicore::Tensor::empty(shape, dtype, device);
}

} // namespace

void DeepseekV4DecoderLayerSharedScratch::preallocate_scratch(size_t hidden_size,
                                                              size_t hc_mult,
                                                              infinicore::DataType dtype,
                                                              const infinicore::Device &device) {
    if (can_use_scratch_tensor(max_attn_in, {kMaxDecodeTokens, hidden_size}, dtype, device)
        && can_use_scratch_tensor(max_attn_post, {kMaxDecodeTokens, hc_mult}, infinicore::DataType::F32, device)
        && can_use_scratch_tensor(max_attn_comb, {kMaxDecodeTokens, hc_mult, hc_mult}, infinicore::DataType::F32, device)
        && can_use_scratch_tensor(max_ffn_in, {kMaxDecodeTokens, hidden_size}, dtype, device)
        && can_use_scratch_tensor(max_ffn_post, {kMaxDecodeTokens, hc_mult}, infinicore::DataType::F32, device)
        && can_use_scratch_tensor(max_ffn_comb, {kMaxDecodeTokens, hc_mult, hc_mult}, infinicore::DataType::F32, device)) {
        return;
    }

    max_attn_in = infinicore::Tensor::empty({kMaxDecodeTokens, hidden_size}, dtype, device);
    max_attn_post = infinicore::Tensor::empty({kMaxDecodeTokens, hc_mult}, infinicore::DataType::F32, device);
    max_attn_comb = infinicore::Tensor::empty({kMaxDecodeTokens, hc_mult, hc_mult}, infinicore::DataType::F32, device);
    max_ffn_in = infinicore::Tensor::empty({kMaxDecodeTokens, hidden_size}, dtype, device);
    max_ffn_post = infinicore::Tensor::empty({kMaxDecodeTokens, hc_mult}, infinicore::DataType::F32, device);
    max_ffn_comb = infinicore::Tensor::empty({kMaxDecodeTokens, hc_mult, hc_mult}, infinicore::DataType::F32, device);
    assert(max_attn_in && max_attn_post && max_attn_comb && max_ffn_in
           && max_ffn_post && max_ffn_comb);
}

infinicore::Tensor DeepseekV4DecoderLayerSharedScratch::get_attn_in(const infinicore::Shape &shape,
                                                                    infinicore::DataType dtype,
                                                                    const infinicore::Device &device) const {
    return get_scratch_or_empty(max_attn_in, shape, dtype, device);
}

infinicore::Tensor DeepseekV4DecoderLayerSharedScratch::get_attn_post(const infinicore::Shape &shape,
                                                                      infinicore::DataType dtype,
                                                                      const infinicore::Device &device) const {
    return get_scratch_or_empty(max_attn_post, shape, dtype, device);
}

infinicore::Tensor DeepseekV4DecoderLayerSharedScratch::get_attn_comb(const infinicore::Shape &shape,
                                                                      infinicore::DataType dtype,
                                                                      const infinicore::Device &device) const {
    return get_scratch_or_empty(max_attn_comb, shape, dtype, device);
}

infinicore::Tensor DeepseekV4DecoderLayerSharedScratch::get_ffn_in(const infinicore::Shape &shape,
                                                                   infinicore::DataType dtype,
                                                                   const infinicore::Device &device) const {
    return get_scratch_or_empty(max_ffn_in, shape, dtype, device);
}

infinicore::Tensor DeepseekV4DecoderLayerSharedScratch::get_ffn_post(const infinicore::Shape &shape,
                                                                     infinicore::DataType dtype,
                                                                     const infinicore::Device &device) const {
    return get_scratch_or_empty(max_ffn_post, shape, dtype, device);
}

infinicore::Tensor DeepseekV4DecoderLayerSharedScratch::get_ffn_comb(const infinicore::Shape &shape,
                                                                     infinicore::DataType dtype,
                                                                     const infinicore::Device &device) const {
    return get_scratch_or_empty(max_ffn_comb, shape, dtype, device);
}

void DeepseekV4SharedExpertScratch::preallocate_scratch(size_t intermediate_size_per_partition,
                                                        infinicore::DataType dtype,
                                                        const infinicore::Device &device) {
    if (can_use_scratch_tensor(max_gate_up, {kMaxDecodeTokens, intermediate_size_per_partition * 2}, dtype, device)
        && can_use_scratch_tensor(max_activated, {kMaxDecodeTokens, intermediate_size_per_partition}, dtype, device)) {
        return;
    }

    max_gate_up = infinicore::Tensor::empty({kMaxDecodeTokens, intermediate_size_per_partition * 2}, dtype, device);
    max_activated = infinicore::Tensor::empty({kMaxDecodeTokens, intermediate_size_per_partition}, dtype, device);
    assert(max_gate_up && max_activated);
}

infinicore::Tensor DeepseekV4SharedExpertScratch::get_gate_up(const infinicore::Shape &shape,
                                                              infinicore::DataType dtype,
                                                              const infinicore::Device &device) const {
    return get_scratch_or_empty(max_gate_up, shape, dtype, device);
}

infinicore::Tensor DeepseekV4SharedExpertScratch::get_activated(const infinicore::Shape &shape,
                                                                infinicore::DataType dtype,
                                                                const infinicore::Device &device) const {
    return get_scratch_or_empty(max_activated, shape, dtype, device);
}

void DeepseekV4AttentionScratch::preallocate_attn_out(size_t num_local_attention_heads,
                                                      size_t head_dim,
                                                      infinicore::DataType dtype,
                                                      const infinicore::Device &device) {
    if (can_use_scratch_tensor(max_attn_out, {kMaxDecodeTokens, num_local_attention_heads, head_dim}, dtype, device)) {
        return;
    }

    max_attn_out = infinicore::Tensor::empty({kMaxDecodeTokens, num_local_attention_heads, head_dim}, dtype, device);
    assert(max_attn_out);
}

infinicore::Tensor DeepseekV4AttentionScratch::get_attn_out(const infinicore::Shape &shape,
                                                            infinicore::DataType dtype,
                                                            const infinicore::Device &device) const {
    return get_scratch_or_empty(max_attn_out, shape, dtype, device);
}

void DeepseekV4RoutedExpertScratch::preallocate_scratch(size_t hidden_size,
                                                        infinicore::DataType dtype,
                                                        const infinicore::Device &device) {
    if (can_use_scratch_tensor(max_output, {kMaxDecodeTokens, hidden_size}, dtype, device)
        && can_use_scratch_tensor(max_contiguous_hidden, {kMaxDecodeTokens, hidden_size}, dtype, device)) {
        return;
    }

    max_output = infinicore::Tensor::empty({kMaxDecodeTokens, hidden_size}, dtype, device);
    max_contiguous_hidden = infinicore::Tensor::empty({kMaxDecodeTokens, hidden_size}, dtype, device);
    assert(max_output && max_contiguous_hidden);
}

infinicore::Tensor DeepseekV4RoutedExpertScratch::get_output(const infinicore::Shape &shape,
                                                             infinicore::DataType dtype,
                                                             const infinicore::Device &device) const {
    return get_scratch_or_empty(max_output, shape, dtype, device);
}

infinicore::Tensor DeepseekV4RoutedExpertScratch::get_contiguous_hidden(const infinicore::Shape &shape,
                                                                        infinicore::DataType dtype,
                                                                        const infinicore::Device &device) const {
    return get_scratch_or_empty(max_contiguous_hidden, shape, dtype, device);
}

} // namespace infinilm::models::deepseek_v4
