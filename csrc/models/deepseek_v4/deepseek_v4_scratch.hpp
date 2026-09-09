#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>

namespace infinilm::models::deepseek_v4 {

struct DeepseekV4DecoderLayerSharedScratch {
    static constexpr size_t kMaxDecodeTokens = 33;

    infinicore::Tensor max_attn_in;
    infinicore::Tensor max_attn_residual;
    infinicore::Tensor max_attn_post;
    infinicore::Tensor max_attn_comb;
    infinicore::Tensor max_ffn_in;
    infinicore::Tensor max_ffn_residual;
    infinicore::Tensor max_ffn_post;
    infinicore::Tensor max_ffn_comb;

    void preallocate_scratch(size_t hidden_size,
                             size_t hc_mult,
                             infinicore::DataType dtype,
                             const infinicore::Device &device);

    infinicore::Tensor get_attn_in(const infinicore::Shape &shape,
                                   infinicore::DataType dtype,
                                   const infinicore::Device &device) const;
    infinicore::Tensor get_attn_residual(const infinicore::Shape &shape,
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
    infinicore::Tensor get_ffn_residual(const infinicore::Shape &shape,
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
                           const infinicore::Device &device);

private:
    static size_t numel(const infinicore::Shape &shape);

    infinicore::Tensor buffer_;
    size_t capacity_{0};
};

} // namespace infinilm::models::deepseek_v4
