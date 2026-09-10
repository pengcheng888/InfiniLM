#include "deepseek_v4_rope.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

double yarn_correction_dim(double num_rotations, size_t rotary_dim, double base, size_t original_max_position) {
    return (static_cast<double>(rotary_dim) * std::log(static_cast<double>(original_max_position) / (num_rotations * kTwoPi))) / (2.0 * std::log(base));
}

std::pair<size_t, size_t> yarn_correction_range(double beta_fast,
                                                double beta_slow,
                                                size_t rotary_dim,
                                                double base,
                                                size_t original_max_position) {
    const double low = std::floor(yarn_correction_dim(beta_fast, rotary_dim, base, original_max_position));
    const double high = std::ceil(yarn_correction_dim(beta_slow, rotary_dim, base, original_max_position));
    const auto max_idx = static_cast<double>(rotary_dim - 1);
    const size_t low_idx = static_cast<size_t>(std::clamp(low, 0.0, max_idx));
    const size_t high_idx = static_cast<size_t>(std::clamp(high, 0.0, max_idx));
    return {low_idx, high_idx};
}

double yarn_linear_ramp(size_t dim, size_t low, size_t high) {
    double min_v = static_cast<double>(low);
    double max_v = static_cast<double>(high);
    if (min_v == max_v) {
        max_v += 0.001;
    }
    return std::clamp((static_cast<double>(dim) - min_v) / (max_v - min_v), 0.0, 1.0);
}

} // namespace

infinicore::Tensor build_deepseek_v4_rope_freqs_cis(size_t qk_rope_head_dim,
                                                    size_t max_position_embeddings,
                                                    bool use_compress_rope,
                                                    double rope_theta,
                                                    double compress_rope_theta,
                                                    double rope_factor,
                                                    double rope_beta_fast,
                                                    double rope_beta_slow,
                                                    size_t rope_original_seq_len,
                                                    const infinicore::Device &device) {
    if (qk_rope_head_dim == 0 || qk_rope_head_dim % 2 != 0 || max_position_embeddings == 0) {
        throw std::runtime_error("DeepSeekV4 RoPE: invalid configuration");
    }

    const size_t half_dim = qk_rope_head_dim / 2;
    const size_t numel = max_position_embeddings * qk_rope_head_dim;
    const double rope_base = use_compress_rope ? compress_rope_theta : rope_theta;
    const size_t original_seq_len = use_compress_rope ? rope_original_seq_len : 0;

    std::vector<double> inv_freq(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        inv_freq[i] = 1.0 / std::pow(rope_base, static_cast<double>(2 * i) / static_cast<double>(qk_rope_head_dim));
    }

    if (original_seq_len > 0 && rope_factor != 1.0) {
        auto [low, high] = yarn_correction_range(rope_beta_fast,
                                                 rope_beta_slow,
                                                 qk_rope_head_dim,
                                                 rope_base,
                                                 original_seq_len);
        for (size_t i = 0; i < half_dim; ++i) {
            const double smooth = 1.0 - yarn_linear_ramp(i, low, high);
            inv_freq[i] = (inv_freq[i] / rope_factor) * (1.0 - smooth) + inv_freq[i] * smooth;
        }
    }

    std::vector<float> freqs_data(numel);
    for (size_t pos = 0; pos < max_position_embeddings; ++pos) {
        for (size_t i = 0; i < half_dim; ++i) {
            const double angle = static_cast<double>(pos) * inv_freq[i];
            const size_t offset = pos * qk_rope_head_dim + 2 * i;
            freqs_data[offset] = static_cast<float>(std::cos(angle));
            freqs_data[offset + 1] = static_cast<float>(std::sin(angle));
        }
    }

    auto freqs_cache = infinicore::Tensor::empty({max_position_embeddings, qk_rope_head_dim}, infinicore::DataType::F32, device);
    const auto cpu = infinicore::Device::cpu();
    auto freqs_cpu = infinicore::Tensor::from_blob(freqs_data.data(), {max_position_embeddings, qk_rope_head_dim}, infinicore::DataType::F32, cpu);
    freqs_cache->copy_from(freqs_cpu);
    return freqs_cache;
}

} // namespace infinilm::models::deepseek_v4
