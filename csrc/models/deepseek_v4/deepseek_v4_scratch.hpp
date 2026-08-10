#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>

namespace infinilm::models::deepseek_v4 {

struct DeepseekV4DecoderLayerSharedScratch {
    static constexpr size_t kMaxDecodeTokens = 33;

    infinicore::Tensor max_attn_in;
    infinicore::Tensor max_attn_post;
    infinicore::Tensor max_attn_comb;
    infinicore::Tensor max_ffn_in;
    infinicore::Tensor max_ffn_post;
    infinicore::Tensor max_ffn_comb;

    void preallocate_scratch(size_t hidden_size,
                             size_t hc_mult,
                             infinicore::DataType dtype,
                             const infinicore::Device &device);

    infinicore::Tensor get_attn_in(const infinicore::Shape &shape,
                                   infinicore::DataType dtype,
                                   const infinicore::Device &device) const;
    infinicore::Tensor get_attn_post(const infinicore::Shape &shape,
                                     infinicore::DataType dtype,
                                     const infinicore::Device &device) const;
    infinicore::Tensor get_attn_comb(const infinicore::Shape &shape,
                                     infinicore::DataType dtype,
                                     const infinicore::Device &device) const;
    infinicore::Tensor get_ffn_in(const infinicore::Shape &shape,
                                  infinicore::DataType dtype,
                                  const infinicore::Device &device) const;
    infinicore::Tensor get_ffn_post(const infinicore::Shape &shape,
                                    infinicore::DataType dtype,
                                    const infinicore::Device &device) const;
    infinicore::Tensor get_ffn_comb(const infinicore::Shape &shape,
                                    infinicore::DataType dtype,
                                    const infinicore::Device &device) const;
};

class DeepseekV4FlatScratchBuffer {
public:
    infinicore::Tensor get(const infinicore::Shape &shape,
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

    size_t capacity() const {
        return capacity_;
    }

private:
    static size_t numel(const infinicore::Shape &shape) {
        size_t result = 1;
        for (const auto dim : shape) {
            result *= dim;
        }
        return result;
    }

    infinicore::Tensor buffer_;
    size_t capacity_{0};
};

class DeepseekV4SharedExpertScratch {
public:
    static constexpr size_t kMaxDecodeTokens = DeepseekV4DecoderLayerSharedScratch::kMaxDecodeTokens;

    void preallocate_scratch(size_t intermediate_size_per_partition,
                             infinicore::DataType dtype,
                             const infinicore::Device &device);

    infinicore::Tensor get_gate_up(const infinicore::Shape &shape,
                                   infinicore::DataType dtype,
                                   const infinicore::Device &device) const;

    infinicore::Tensor get_activated(const infinicore::Shape &shape,
                                     infinicore::DataType dtype,
                                     const infinicore::Device &device) const;

private:
    infinicore::Tensor max_gate_up;
    infinicore::Tensor max_activated;
};

class DeepseekV4AttentionScratch {
public:
    static constexpr size_t kMaxDecodeTokens = DeepseekV4DecoderLayerSharedScratch::kMaxDecodeTokens;

    void preallocate_attn_out(size_t num_local_attention_heads,
                              size_t head_dim,
                              infinicore::DataType dtype,
                              const infinicore::Device &device);

    infinicore::Tensor get_attn_out(const infinicore::Shape &shape,
                                    infinicore::DataType dtype,
                                    const infinicore::Device &device) const;

private:
    infinicore::Tensor max_attn_out;
};

class DeepseekV4MLAScratch {
public:
    infinicore::Tensor get_lse(const infinicore::Shape &shape,
                               infinicore::DataType dtype,
                               const infinicore::Device &device) {
        return lse_.get(shape, dtype, device);
    }

    infinicore::Tensor get_lse_accum(const infinicore::Shape &shape,
                                     infinicore::DataType dtype,
                                     const infinicore::Device &device) {
        return lse_accum_.get(shape, dtype, device);
    }

    infinicore::Tensor get_o_accum(const infinicore::Shape &shape,
                                   infinicore::DataType dtype,
                                   const infinicore::Device &device) {
        return o_accum_.get(shape, dtype, device);
    }

private:
    DeepseekV4FlatScratchBuffer lse_;
    DeepseekV4FlatScratchBuffer lse_accum_;
    DeepseekV4FlatScratchBuffer o_accum_;
};

class DeepseekV4RoutedExpertScratch {
public:
    static constexpr size_t kMaxDecodeTokens = DeepseekV4DecoderLayerSharedScratch::kMaxDecodeTokens;

    void preallocate_scratch(size_t hidden_size,
                             infinicore::DataType dtype,
                             const infinicore::Device &device);

    infinicore::Tensor get_output(const infinicore::Shape &shape,
                                  infinicore::DataType dtype,
                                  const infinicore::Device &device) const;

    infinicore::Tensor get_fused_output(const infinicore::Shape &shape,
                                        infinicore::DataType dtype,
                                        const infinicore::Device &device) const;

private:
    infinicore::Tensor max_output;
    infinicore::Tensor max_fused_output;
};

} // namespace infinilm::models::deepseek_v4
