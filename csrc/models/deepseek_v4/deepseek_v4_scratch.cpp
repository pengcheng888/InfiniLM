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
        && can_use_scratch_tensor(max_attn_residual, {kMaxDecodeTokens, hc_mult, hidden_size}, dtype, device)
        && can_use_scratch_tensor(max_attn_post, {kMaxDecodeTokens, hc_mult}, infinicore::DataType::F32, device)
        && can_use_scratch_tensor(max_attn_comb, {kMaxDecodeTokens, hc_mult, hc_mult}, infinicore::DataType::F32, device)
        && can_use_scratch_tensor(max_ffn_in, {kMaxDecodeTokens, hidden_size}, dtype, device)
        && can_use_scratch_tensor(max_ffn_residual, {kMaxDecodeTokens, hc_mult, hidden_size}, dtype, device)
        && can_use_scratch_tensor(max_ffn_post, {kMaxDecodeTokens, hc_mult}, infinicore::DataType::F32, device)
        && can_use_scratch_tensor(max_ffn_comb, {kMaxDecodeTokens, hc_mult, hc_mult}, infinicore::DataType::F32, device)) {
        return;
    }

    max_attn_in = infinicore::Tensor::empty({kMaxDecodeTokens, hidden_size}, dtype, device);
    max_attn_residual = infinicore::Tensor::empty({kMaxDecodeTokens, hc_mult, hidden_size}, dtype, device);
    max_attn_post = infinicore::Tensor::empty({kMaxDecodeTokens, hc_mult}, infinicore::DataType::F32, device);
    max_attn_comb = infinicore::Tensor::empty({kMaxDecodeTokens, hc_mult, hc_mult}, infinicore::DataType::F32, device);
    max_ffn_in = infinicore::Tensor::empty({kMaxDecodeTokens, hidden_size}, dtype, device);
    max_ffn_residual = infinicore::Tensor::empty({kMaxDecodeTokens, hc_mult, hidden_size}, dtype, device);
    max_ffn_post = infinicore::Tensor::empty({kMaxDecodeTokens, hc_mult}, infinicore::DataType::F32, device);
    max_ffn_comb = infinicore::Tensor::empty({kMaxDecodeTokens, hc_mult, hc_mult}, infinicore::DataType::F32, device);
    assert(max_attn_in && max_attn_residual && max_attn_post && max_attn_comb
           && max_ffn_in && max_ffn_residual && max_ffn_post && max_ffn_comb);
}

infinicore::Tensor DeepseekV4DecoderLayerSharedScratch::get_attn_in(const infinicore::Shape &shape,
                                                                    infinicore::DataType dtype,
                                                                    const infinicore::Device &device) const {
    return get_scratch_or_empty(max_attn_in, shape, dtype, device);
}

infinicore::Tensor DeepseekV4DecoderLayerSharedScratch::get_attn_residual(const infinicore::Shape &shape,
                                                                          infinicore::DataType dtype,
                                                                          const infinicore::Device &device) const {
    return get_scratch_or_empty(max_attn_residual, shape, dtype, device);
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

infinicore::Tensor DeepseekV4DecoderLayerSharedScratch::get_ffn_residual(const infinicore::Shape &shape,
                                                                         infinicore::DataType dtype,
                                                                         const infinicore::Device &device) const {
    return get_scratch_or_empty(max_ffn_residual, shape, dtype, device);
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

infinicore::Tensor DeepseekV4FlatScratchBuffer::get(const infinicore::Shape &shape,
                                                    infinicore::DataType dtype,
                                                    const infinicore::Device &device) {
    const size_t required = numel(shape);
    if (!buffer_ || capacity_ < required || buffer_->dtype() != dtype || buffer_->device() != device) {
        buffer_ = infinicore::Tensor::empty({required}, dtype, device);
        capacity_ = required;
    }
    if (buffer_->numel() == required) {
        return buffer_->view(shape);
    }
    return buffer_->narrow({{0, 0, required}})->view(shape);
}

size_t DeepseekV4FlatScratchBuffer::numel(const infinicore::Shape &shape) {
    size_t result = 1;
    for (const auto dim : shape) {
        result *= dim;
    }
    return result;
}

} // namespace infinilm::models::deepseek_v4
