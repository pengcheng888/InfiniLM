#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>

namespace infinilm::models::deepseek_v4 {

infinicore::Tensor build_deepseek_v4_rope_freqs_cis(size_t qk_rope_head_dim,
                                                    size_t max_position_embeddings,
                                                    bool use_compress_rope,
                                                    double rope_theta,
                                                    double compress_rope_theta,
                                                    double rope_factor,
                                                    double rope_beta_fast,
                                                    double rope_beta_slow,
                                                    size_t rope_original_seq_len,
                                                    const infinicore::Device &device);

} // namespace infinilm::models::deepseek_v4
