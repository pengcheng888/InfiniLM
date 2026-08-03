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

DeepseekV4Compressor::DeepseekV4Compressor(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t head_dim,
    size_t compress_ratio,
    const infinicore::Device &device)
    : head_dim_(head_dim),
      compress_ratio_(compress_ratio) {
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    if (compress_ratio_ == 4) {
        proj_size_ = 2 * head_dim_;
        page_size_ = kDsv4C4PageSize;
        INFINICORE_NN_PARAMETER_INIT(ape, ({kDsv4C4PageSize / 16, proj_size_}, infinicore::DataType::F32, device));
    } else if (compress_ratio_ == 128) {
        proj_size_ = head_dim_;
        page_size_ = kDsv4C128PageSize;
        INFINICORE_NN_PARAMETER_INIT(ape, ({128, proj_size_}, infinicore::DataType::F32, device));
    } else {
        throw std::runtime_error("DeepseekV4Compressor: unsupported compress_ratio");
    }

    wgate_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wgate", hidden_size, proj_size_, false, dtype, device);
    wkv_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wkv", hidden_size, proj_size_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, head_dim_, rms_norm_eps, dtype, device);
}

void DeepseekV4Compressor::process_weights_after_loading() {
    wgate_->process_weights_after_loading();
    wkv_->process_weights_after_loading();
}

void DeepseekV4Compressor::reset_runtime_state() const {
    wgate_->reset_runtime_state();
    wkv_->reset_runtime_state();
}

infinicore::Tensor DeepseekV4Compressor::forward_kv_score(const infinicore::Tensor &hidden_states) const {
    auto x0 = hidden_states;
    auto kv = wkv_->forward(x0);
    auto x1 = hidden_states;
    auto gate = wgate_->forward(x1);
    return infinicore::op::cat({kv, gate}, -1);
}

void DeepseekV4Compressor::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &pos_ids,
    size_t seq_len,
    const infinicore::Tensor &rope_freqs_cis,
    const infinicore::Tensor &compressor_state,
    const infinicore::Tensor &cache_raw,
    const infinicore::Tensor &out_loc,
    const infinicore::Tensor &compress_positions,
    const infinicore::Tensor &write_loc,
    std::optional<infinicore::Tensor> extra_loc) const {
    {
        const auto event = compress_ratio_ == 4 ? profile::Event::AttentionC4Compress : profile::Event::AttentionC128Compress;
        profile::ScopedTimer timer(event, seq_len);
        auto kv_score = forward_kv_score(hidden_states);
        const auto expected_score_dim = 2 * proj_size_;
        if (kv_score->ndim() != 2 || kv_score->size(1) != expected_score_dim) {
            throw std::runtime_error("DeepseekV4Compressor::forward compressor output shape mismatch");
        }

        infinicore::Tensor compressed_kv;
        if (compress_ratio_ == 4) {
            if (!extra_loc) {
                throw std::runtime_error("DeepseekV4Compressor::forward requires extra_loc for C4 compression");
            }
            compressed_kv = infinicore::op::deepseek_v4_c4_compress_stateful(kv_score,
                                                                             ape_,
                                                                             compressor_state,
                                                                             write_loc,
                                                                             extra_loc.value(),
                                                                             pos_ids);
        } else if (compress_ratio_ == 128) {
            compressed_kv = infinicore::op::deepseek_v4_c128_compress_stateful(kv_score,
                                                                               ape_,
                                                                               compressor_state,
                                                                               write_loc,
                                                                               pos_ids);
        } else {
            throw std::runtime_error("DeepseekV4Compressor::forward found unsupported compress_ratio");
        }

        infinicore::op::deepseek_v4_compress_fused_norm_rope_(compressed_kv,
                                                              norm_->weight(),
                                                              norm_->eps(),
                                                              rope_freqs_cis,
                                                              compress_positions);

        infinicore::op::deepseek_v4_store_flashmla_raw_cache_(compressed_kv,
                                                              cache_raw,
                                                              out_loc,
                                                              static_cast<int>(page_size_));
    }
}

} // namespace infinilm::models::deepseek_v4
