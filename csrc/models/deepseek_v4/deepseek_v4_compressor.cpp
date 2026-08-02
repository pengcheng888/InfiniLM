#include "deepseek_v4_compressor.hpp"

#include "deepseek_v4_profile.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_cache.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
namespace {

constexpr size_t kDsv4C4PageSize = 64;
constexpr size_t kDsv4C128PageSize = 2;

} // namespace

DeepseekV4CSACompressor::DeepseekV4CSACompressor(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t head_dim,
    const infinicore::Device &device)
    : head_dim_(head_dim) {
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t proj_size = 2 * head_dim_;
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_PARAMETER_INIT(ape, ({kDsv4C4PageSize / 16, proj_size}, infinicore::DataType::F32, device));
    wgate_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wgate", hidden_size, proj_size, false, dtype, device);
    wkv_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wkv", hidden_size, proj_size, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, head_dim_, rms_norm_eps, dtype, device);
}

void DeepseekV4CSACompressor::process_weights_after_loading() {
    wgate_->process_weights_after_loading();
    wkv_->process_weights_after_loading();
}

void DeepseekV4CSACompressor::reset_runtime_state() const {
    wgate_->reset_runtime_state();
    wkv_->reset_runtime_state();
}

infinicore::Tensor DeepseekV4CSACompressor::forward_kv_score(const infinicore::Tensor &hidden_states) const {
    auto x0 = hidden_states;
    auto kv = wkv_->forward(x0);
    auto x1 = hidden_states;
    auto gate = wgate_->forward(x1);
    return infinicore::op::cat({kv, gate}, -1);
}

void DeepseekV4CSACompressor::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &pos_ids,
    size_t seq_len,
    const infinicore::Tensor &rope_freqs_cis,
    const infinicore::Tensor &compressor_state,
    const infinicore::Tensor &c4_cache_raw,
    const infinicore::Tensor &c4_out_loc,
    const infinicore::Tensor &c4_positions,
    const infinicore::Tensor &c4_write_loc,
    const infinicore::Tensor &c4_extra_loc) const {
    {
        profile::ScopedTimer timer(profile::Event::AttentionC4Compress, seq_len);
        auto c4_kv_score = forward_kv_score(hidden_states);
        if (c4_kv_score->ndim() != 2 || c4_kv_score->size(1) < 2 * head_dim_) {
            throw std::runtime_error("DeepseekV4CSACompressor::forward C4 compressor output shape mismatch");
        }
        auto c4_kv = infinicore::op::deepseek_v4_c4_compress_stateful(c4_kv_score,
                                                                      ape_,
                                                                      compressor_state,
                                                                      c4_write_loc,
                                                                      c4_extra_loc,
                                                                      pos_ids);
        infinicore::op::deepseek_v4_compress_fused_norm_rope_(c4_kv,
                                                              norm_->weight(),
                                                              norm_->eps(),
                                                              rope_freqs_cis,
                                                              c4_positions);

        infinicore::op::deepseek_v4_store_flashmla_raw_cache_(c4_kv,
                                                              c4_cache_raw,
                                                              c4_out_loc,
                                                              static_cast<int>(kDsv4C4PageSize));
    }
}

DeepseekV4HCACompressor::DeepseekV4HCACompressor(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t head_dim,
    const infinicore::Device &device)
    : head_dim_(head_dim) {
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t proj_size = head_dim_;
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_PARAMETER_INIT(ape, ({128, proj_size}, infinicore::DataType::F32, device));
    wgate_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wgate", hidden_size, proj_size, false, dtype, device);
    wkv_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wkv", hidden_size, proj_size, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, head_dim_, rms_norm_eps, dtype, device);
}

void DeepseekV4HCACompressor::process_weights_after_loading() {
    wgate_->process_weights_after_loading();
    wkv_->process_weights_after_loading();
}

void DeepseekV4HCACompressor::reset_runtime_state() const {
    wgate_->reset_runtime_state();
    wkv_->reset_runtime_state();
}

infinicore::Tensor DeepseekV4HCACompressor::forward_kv_score(const infinicore::Tensor &hidden_states) const {
    auto x0 = hidden_states;
    auto kv = wkv_->forward(x0);
    auto x1 = hidden_states;
    auto gate = wgate_->forward(x1);
    return infinicore::op::cat({kv, gate}, -1);
}

void DeepseekV4HCACompressor::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &pos_ids,
    size_t seq_len,
    const infinicore::Tensor &rope_freqs_cis,
    const infinicore::Tensor &compressor_state,
    const infinicore::Tensor &c128_cache_raw,
    const infinicore::Tensor &c128_out_loc,
    const infinicore::Tensor &c128_positions,
    const infinicore::Tensor &c128_write_loc) const {
    {
        profile::ScopedTimer timer(profile::Event::AttentionC128Compress, seq_len);
        auto c128_kv_score = forward_kv_score(hidden_states);
        if (c128_kv_score->ndim() != 2 || c128_kv_score->size(1) != 2 * head_dim_) {
            throw std::runtime_error("DeepseekV4HCACompressor::forward C128 compressor output shape mismatch");
        }
        auto c128_kv = infinicore::op::deepseek_v4_c128_compress_stateful(c128_kv_score,
                                                                          ape_,
                                                                          compressor_state,
                                                                          c128_write_loc,
                                                                          pos_ids);

        infinicore::op::deepseek_v4_compress_fused_norm_rope_(c128_kv,
                                                              norm_->weight(),
                                                              norm_->eps(),
                                                              rope_freqs_cis,
                                                              c128_positions);

        infinicore::op::deepseek_v4_store_flashmla_raw_cache_(c128_kv,
                                                              c128_cache_raw,
                                                              c128_out_loc,
                                                              static_cast<int>(kDsv4C128PageSize));
    }
}

} // namespace infinilm::models::deepseek_v4
