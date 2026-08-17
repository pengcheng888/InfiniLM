#include "deepseek_v4_compressor.hpp"

#include "deepseek_v4_profile.hpp"
#include "infinicore/ops/deepseek_v4_compress_norm_rope_store.hpp"
#include "infinicore/ops/deepseek_v4_compress_sglang_stateful.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_cache.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"
#include "infinicore/ops/deepseek_v4_indexer_compress_norm_rope_store.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
namespace {

constexpr size_t kDsv4C4PageSize = 64;
constexpr size_t kDsv4C128PageSize = 2;

void check_sglang_ape_shape(const infinicore::Tensor &ape, size_t compress_ratio, size_t head_dim) {
    if (compress_ratio == 4) {
        if (ape->ndim() != 2 || ape->size(0) != 8 || ape->size(1) != head_dim) {
            throw std::runtime_error("DeepseekV4Compressor::compress_forward C4 sglang compress expects ape [8, head_dim]");
        }
        return;
    }
    if (compress_ratio == 128) {
        if (ape->ndim() != 2 || ape->size(0) < 128 || ape->size(1) != head_dim) {
            throw std::runtime_error("DeepseekV4Compressor::compress_forward C128 sglang compress expects ape [128, head_dim]");
        }
        return;
    }
}

} // namespace

DeepseekV4Compressor::DeepseekV4Compressor(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t head_dim,
    size_t compress_ratio,
    const infinicore::Device &device,
    DeepseekV4CompressorStoreKind store_kind)
    : head_dim_(head_dim),
      compress_ratio_(compress_ratio),
      store_kind_(store_kind) {
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    if (compress_ratio_ == 4) {
        proj_size_ = 2 * head_dim_;
        page_size_ = kDsv4C4PageSize;
        INFINICORE_NN_PARAMETER_INIT(ape, ({8, head_dim_}, infinicore::DataType::F32, device));
    } else if (compress_ratio_ == 128) {
        proj_size_ = head_dim_;
        page_size_ = kDsv4C128PageSize;
        INFINICORE_NN_PARAMETER_INIT(ape, ({128, proj_size_}, infinicore::DataType::F32, device)); // F32
    } else {
        throw std::runtime_error("DeepseekV4Compressor: unsupported compress_ratio");
    }

    auto register_fn = [this](const std::string &name, infinicore::nn::Parameter param) {
        this->register_parameter(name, std::move(param));
    };
    wkv_gate_ = std::make_shared<infinilm::layers::linear::FusedReplicatedLinear>(
        hidden_size,
        proj_size_,
        "wkv",
        "wgate",
        register_fn,
        dtype,
        device);                                                             // BF16
    INFINICORE_NN_MODULE_INIT(norm, head_dim_, rms_norm_eps, dtype, device); // BF16
}

void DeepseekV4Compressor::process_weights_after_loading() {
    wkv_gate_->process_weights_after_loading();
}

void DeepseekV4Compressor::reset_runtime_state() const {
    wkv_gate_->reset_runtime_state();
}

infinicore::Tensor DeepseekV4Compressor::compute_kv_score(const infinicore::Tensor &hidden_states) const {
    return wkv_gate_->forward(hidden_states);
}

infinicore::Tensor DeepseekV4Compressor::compress_forward(
    const infinicore::Tensor &kv_score,
    const infinicore::Tensor &compressor_state,
    const infinicore::Tensor &write_loc,
    std::optional<infinicore::Tensor> extra_loc,
    const infinicore::Tensor &pos_ids) const {
    infinicore::Tensor kv_compressed;

    bool use_compress_sglang = true;
    if (compress_ratio_ == 4) {
        if (!extra_loc) {
            throw std::runtime_error("DeepseekV4Compressor::compress_forward requires extra_loc for C4 compression");
        }
        if (use_compress_sglang) {
            check_sglang_ape_shape(ape_, compress_ratio_, head_dim_);
            kv_compressed = infinicore::op::deepseek_v4_c4_compress_sglang_stateful(kv_score,
                                                                                    ape_,
                                                                                    compressor_state,
                                                                                    write_loc,
                                                                                    extra_loc.value(),
                                                                                    pos_ids);
        } else {
            kv_compressed = infinicore::op::deepseek_v4_c4_compress_stateful(kv_score,
                                                                             ape_,
                                                                             compressor_state,
                                                                             write_loc,
                                                                             extra_loc.value(),
                                                                             pos_ids);
        }
    } else if (compress_ratio_ == 128) {
        if (use_compress_sglang) {
            check_sglang_ape_shape(ape_, compress_ratio_, head_dim_);
            kv_compressed = infinicore::op::deepseek_v4_c128_compress_sglang_stateful(kv_score,
                                                                                      ape_,
                                                                                      compressor_state,
                                                                                      write_loc,
                                                                                      pos_ids);
        } else {
            kv_compressed = infinicore::op::deepseek_v4_c128_compress_stateful(kv_score,
                                                                               ape_,
                                                                               compressor_state,
                                                                               write_loc,
                                                                               pos_ids);
        }
    } else {
        throw std::runtime_error("DeepseekV4Compressor::compress_forward found unsupported compress_ratio");
    }
    return kv_compressed;
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
    const int repeats = 1; // 1000   total_ms=60
    for (int i = 0; i < repeats; ++i) {
        const auto event = compress_ratio_ == 4 ? profile::Event::AttentionC4Compress : profile::Event::AttentionC128Compress;
        profile::ScopedTimer timer(event, seq_len);
        infinicore::Tensor kv_score;
        {
            kv_score = compute_kv_score(hidden_states);
        }

        const auto expected_score_dim = 2 * proj_size_;
        if (kv_score->ndim() != 2 || kv_score->size(1) != expected_score_dim) {
            throw std::runtime_error("DeepseekV4Compressor::forward compressor output shape mismatch");
        }

        // Step 1: compress_forward  这个函数还没有参考sglang去写
        auto kv_compressed = compress_forward(kv_score,
                                              compressor_state,
                                              write_loc,
                                              extra_loc,
                                              pos_ids);
        // Step 2: norm + rope + store. Keep both fused and split FlashMLA paths for analysis.
        {

            switch (store_kind_) {
            case DeepseekV4CompressorStoreKind::FlashMLA: {
                const bool use_fused_flashmla_store = true;
                if (use_fused_flashmla_store) {
                    infinicore::op::deepseek_v4_compress_norm_rope_store_(kv_compressed,
                                                                          norm_->weight(),
                                                                          norm_->eps(),
                                                                          rope_freqs_cis,
                                                                          compress_positions,
                                                                          out_loc,
                                                                          cache_raw,
                                                                          static_cast<int>(page_size_));
                } else {
                    infinicore::op::deepseek_v4_compress_fused_norm_rope_(kv_compressed,
                                                                          norm_->weight(),
                                                                          norm_->eps(),
                                                                          rope_freqs_cis,
                                                                          compress_positions);

                    infinicore::op::deepseek_v4_store_flashmla_raw_cache_(kv_compressed,
                                                                          cache_raw,
                                                                          out_loc,
                                                                          static_cast<int>(page_size_));
                }
                break;
            }
            case DeepseekV4CompressorStoreKind::Indexer: {
                const int repeats = 1; // for test 5000
                for (int i = 0; i < repeats; ++i) {
                    const bool use_fused_indexer_compress_norm_rope_store = true;
                    if (use_fused_indexer_compress_norm_rope_store) {
                        infinicore::op::deepseek_v4_indexer_compress_norm_rope_store_(kv_compressed,
                                                                                      norm_->weight(),
                                                                                      norm_->eps(),
                                                                                      rope_freqs_cis,
                                                                                      compress_positions,
                                                                                      out_loc,
                                                                                      cache_raw,
                                                                                      static_cast<int>(page_size_));
                    } else {
                        infinicore::op::deepseek_v4_compress_fused_norm_rope_(kv_compressed,
                                                                              norm_->weight(),
                                                                              norm_->eps(),
                                                                              rope_freqs_cis,
                                                                              compress_positions);

                        infinicore::op::deepseek_v4_indexer_rotate_(kv_compressed, true);
                        infinicore::op::deepseek_v4_store_indexer_raw_cache_(kv_compressed,
                                                                             cache_raw,
                                                                             out_loc,
                                                                             static_cast<int>(page_size_));
                    }
                }
                break;
            }
            }
        }
    }
}

} // namespace infinilm::models::deepseek_v4
