#include "deepseek_v4_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "deepseek_v4_profile.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_cache.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"
#include "infinicore/ops/deepseek_v4_fused_rope.hpp"
#include "infinicore/ops/deepseek_v4_rms_norm.hpp"
#include "infinicore/ops/deepseek_v4_rmsnorm_self.hpp"
#include "infinicore/ops/deepseek_v4_sparse_attn_indexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {

DeepseekV4Compressor::DeepseekV4Compressor(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t compress_ratio,
                                           size_t compressor_head_dim,
                                           const infinicore::Device &device) {
    if (compress_ratio == 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4Compressor: compress_ratio must be non-zero");
    }
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t coeff = compress_ratio == 4 ? 2 : 1;
    const size_t proj_size = coeff * compressor_head_dim;
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_PARAMETER_INIT(ape, ({compress_ratio, proj_size}, infinicore::DataType::F32, device));
    wgate_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wgate", hidden_size, proj_size, false, dtype, device);
    wkv_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wkv", hidden_size, proj_size, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, compressor_head_dim, rms_norm_eps, dtype, device);
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

DeepseekV4C4Indexer::DeepseekV4C4Indexer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    const auto quantization_method = model_config->get_quantization_method();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t q_lora_rank = model_config->get<size_t>("q_lora_rank");
    index_n_heads_ = model_config->get<size_t>("index_n_heads");
    index_head_dim_ = model_config->get<size_t>("index_head_dim");
    weight_scale_ = static_cast<float>(std::pow(static_cast<double>(index_head_dim_), -0.5) * std::pow(static_cast<double>(index_n_heads_), -0.5));

    wq_b_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wq_b", q_lora_rank, index_n_heads_ * index_head_dim_, quantization_method, false, dtype, device);
    weights_proj_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "weights_proj", hidden_size, index_n_heads_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(compressor, model_config, 4, index_head_dim_, device);
}

void DeepseekV4C4Indexer::process_weights_after_loading() {
    wq_b_->process_weights_after_loading();
    weights_proj_->process_weights_after_loading();
    compressor_->process_weights_after_loading();
}

void DeepseekV4C4Indexer::reset_runtime_state() const {
    wq_b_->reset_runtime_state();
    weights_proj_->reset_runtime_state();
    compressor_->reset_runtime_state();
}

infinicore::Tensor DeepseekV4C4Indexer::compute_q(const infinicore::Tensor &q_lora, size_t seq_len) const {
    auto q_in = q_lora;
    return wq_b_->forward(q_in)->view({seq_len, index_n_heads_, index_head_dim_});
}

infinicore::Tensor DeepseekV4C4Indexer::compute_weights(const infinicore::Tensor &hidden_states) const {
    auto x = hidden_states;
    return weights_proj_->forward(x);
}

infinicore::Tensor DeepseekV4C4Indexer::forward_kv_score(const infinicore::Tensor &hidden_states) const {
    return compressor_->forward_kv_score(hidden_states);
}

DeepseekV4Attention::DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         size_t layer_idx,
                                         const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto dtype = model_config->get_dtype();
    dtype_ = dtype;
    const auto quantization_method = model_config->get_quantization_method();
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    tp_rank_ = tp_rank;
    tp_size_ = tp_size;

    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    num_attention_heads_ = model_config->get<size_t>("num_attention_heads");
    num_key_value_heads_ = model_config->get<size_t>("num_key_value_heads");
    q_lora_rank_ = model_config->get<size_t>("q_lora_rank");
    o_lora_rank_ = model_config->get<size_t>("o_lora_rank");
    qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    index_head_dim_ = model_config->get_or<size_t>("index_head_dim", 128);
    o_groups_ = model_config->get<size_t>("o_groups");
    rms_norm_eps_ = model_config->get<double>("rms_norm_eps");
    max_position_embeddings_ = model_config->get<size_t>("max_position_embeddings");
    rope_theta_ = model_config->get_or<double>("rope_theta", 10000.0);
    compress_rope_theta_ = model_config->get_or<double>("compress_rope_theta", 160000.0);
    if (model_config->get_config_json().contains("rope_scaling") && model_config->get_config_json()["rope_scaling"].is_object()) {
        const auto &rope_scaling = model_config->get_config_json()["rope_scaling"];
        rope_factor_ = rope_scaling.value("factor", 1.0);
        rope_beta_fast_ = rope_scaling.value("beta_fast", 32.0);
        rope_beta_slow_ = rope_scaling.value("beta_slow", 1.0);
        rope_original_seq_len_ = rope_scaling.value("original_max_position_embeddings", 0);
    }

    if (num_key_value_heads_ != 1) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4Attention: num_key_value_heads must be 1");
    }
    if (num_attention_heads_ % tp_size != 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4Attention: num_attention_heads must be divisible by tp_size");
    }
    if (o_groups_ % tp_size != 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4Attention: o_groups must be divisible by tp_size");
    }

    num_local_attention_heads_ = num_attention_heads_ / tp_size;
    num_local_groups_ = o_groups_ / tp_size;

    INFINICORE_NN_PARAMETER_INIT(attn_sink, ({num_attention_heads_}, infinicore::DataType::F32, device));

    wq_a_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wq_a", hidden_size_, q_lora_rank_, quantization_method, false, dtype, device);
    wkv_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wkv", hidden_size_, head_dim_, quantization_method, false, dtype, device);
    wq_b_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "wq_b",
        q_lora_rank_,
        num_attention_heads_ * head_dim_,
        quantization_method,
        false,
        dtype,
        device,
        tp_rank,
        tp_size);

    INFINICORE_NN_MODULE_INIT(q_norm, q_lora_rank_, rms_norm_eps_, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_norm, head_dim_, rms_norm_eps_, dtype, device);

    wo_a_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "wo_a",
        num_attention_heads_ * head_dim_ / o_groups_,
        o_groups_ * o_lora_rank_,
        false,
        dtype,
        device,
        tp_rank,
        tp_size);

    wo_b_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "wo_b",
        o_groups_ * o_lora_rank_,
        hidden_size_,
        quantization_method,
        false,
        dtype,
        device,
        tp_rank,
        tp_size,
        rank_info.comm);

    const auto compress_ratios = model_config->get<std::vector<size_t>>("compress_ratios");
    compress_ratio_ = layer_idx_ < compress_ratios.size() ? compress_ratios[layer_idx_] : 0;
    if (compress_ratio_ != 0) {
        INFINICORE_NN_MODULE_INIT(compressor, model_config, compress_ratio_, head_dim_, device);
        if (compress_ratio_ == 4) {
            INFINICORE_NN_MODULE_INIT(indexer, model_config, device);
        }
    }

    rope_freqs_cis_ = build_rope_freqs_cis(device);
}

namespace {

constexpr size_t kDsv4SwaBlockSize = 256;
constexpr size_t kDsv4C4PageSize = 64;
constexpr size_t kDsv4C128PageSize = 2;
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
    return std::clamp((static_cast<double>(dim) - min_v) / (max_v - min_v),
                      0.0,
                      1.0);
}

} // namespace

infinicore::Tensor DeepseekV4Attention::build_rope_freqs_cis(const infinicore::Device &device) const {
    if (qk_rope_head_dim_ == 0 || qk_rope_head_dim_ % 2 != 0 || max_position_embeddings_ == 0) {
        throw std::runtime_error("DeepseekV4Attention: invalid RoPE configuration");
    }

    const size_t half_dim = qk_rope_head_dim_ / 2;
    const size_t numel = max_position_embeddings_ * qk_rope_head_dim_;
    const double rope_base = compress_ratio_ != 0 ? compress_rope_theta_ : rope_theta_;
    const size_t original_seq_len = compress_ratio_ != 0 ? rope_original_seq_len_ : 0;

    std::vector<double> inv_freq(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        inv_freq[i] = 1.0 / std::pow(rope_base, static_cast<double>(2 * i) / static_cast<double>(qk_rope_head_dim_));
    }

    if (original_seq_len > 0 && rope_factor_ != 1.0) {
        auto [low, high] = yarn_correction_range(rope_beta_fast_,
                                                 rope_beta_slow_,
                                                 qk_rope_head_dim_,
                                                 rope_base,
                                                 original_seq_len);
        for (size_t i = 0; i < half_dim; ++i) {
            const double smooth = 1.0 - yarn_linear_ramp(i, low, high);
            inv_freq[i] = (inv_freq[i] / rope_factor_) * (1.0 - smooth) + inv_freq[i] * smooth;
        }
    }

    std::vector<float> freqs_data(numel);
    for (size_t pos = 0; pos < max_position_embeddings_; ++pos) {
        for (size_t i = 0; i < half_dim; ++i) {
            const double angle = static_cast<double>(pos) * inv_freq[i];
            const size_t offset = pos * qk_rope_head_dim_ + 2 * i;
            freqs_data[offset] = static_cast<float>(std::cos(angle));
            freqs_data[offset + 1] = static_cast<float>(std::sin(angle));
        }
    }

    auto freqs_cache = infinicore::Tensor::empty({max_position_embeddings_, qk_rope_head_dim_}, infinicore::DataType::F32, device);
    const auto cpu = infinicore::Device::cpu();
    auto freqs_cpu = infinicore::Tensor::from_blob(freqs_data.data(), {max_position_embeddings_, qk_rope_head_dim_}, infinicore::DataType::F32, cpu);
    freqs_cache->copy_from(freqs_cpu);
    return freqs_cache;
}

infinicore::Tensor DeepseekV4Attention::attn_out_workspace(size_t seq_len,
                                                           infinicore::DataType dtype,
                                                           const infinicore::Device &device) const {
    const infinicore::Shape expected_shape{seq_len, num_local_attention_heads_, head_dim_};
    if (!attn_out_workspace_ || attn_out_workspace_->shape() != expected_shape || attn_out_workspace_->dtype() != dtype || attn_out_workspace_->device() != device) {
        attn_out_workspace_ = infinicore::Tensor::empty(expected_shape, dtype, device);
    }
    return attn_out_workspace_;
}

void DeepseekV4Attention::apply_rope_(const infinicore::Tensor &positions,
                                      infinicore::Tensor query,
                                      std::optional<infinicore::Tensor> key,
                                      bool inverse) const {
    const auto q_shape = query->shape();
    if (q_shape.empty() || q_shape[0] == 0) {
        return;
    }
    infinicore::op::deepseek_v4_fused_rope_(query, key, rope_freqs_cis_, positions, inverse);
}

namespace {

infinicore::Tensor prepare_dsv4_position_ids(const infinicore::Tensor &position_ids, size_t seq_len) {
    auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        return position_ids->narrow({{0, 0, 1}})->view({seq_len});
    }
    if (pos_shape.size() == 1) {
        return position_ids;
    }
    throw std::runtime_error("DeepseekV4Attention: unexpected position_ids shape");
}

} // namespace

infinicore::Tensor DeepseekV4Attention::forward(const infinicore::Tensor &positions,
                                                const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape();
    if (shape.size() != 2) {
        throw std::runtime_error("DeepseekV4Attention::forward expects hidden_states [tokens, hidden]");
    }
    const size_t seq_len = shape[0];
    profile::ScopedTimer forward_timer(profile::Event::AttentionForward, seq_len);
    auto pos_ids = prepare_dsv4_position_ids(positions, seq_len);

    infinicore::Tensor q_lora;
    infinicore::Tensor q;
    {
        profile::ScopedTimer timer(profile::Event::AttentionQProjection, seq_len);
        auto x0 = hidden_states;
        q_lora = wq_a_->forward(x0);
        q_lora = q_norm_->forward(q_lora);
        auto q_mut = q_lora;
        q = wq_b_->forward(q_mut)->view({seq_len, num_local_attention_heads_, head_dim_});
        q = infinicore::op::deepseek_v4_rmsnorm_self(q, static_cast<float>(rms_norm_eps_));
    }

    infinicore::Tensor kv;
    {
        profile::ScopedTimer timer(profile::Event::AttentionKVProjection, seq_len);
        auto x1 = hidden_states;
        kv = wkv_->forward(x1);
        kv = kv_norm_->forward(kv);
    }

    auto &forward_context = infinilm::global_state::get_forward_context();
    auto *attn_metadata_ptr = &forward_context.attn_metadata;
    auto *dsv4_metadata_ptr = &attn_metadata_ptr->deepseek_v4;
    infinilm::global_state::DeepSeekV4LayerKVCache *layer_cache_ptr = nullptr;
    {
        profile::ScopedTimer timer(profile::Event::AttentionMetadata, seq_len);
        if (!attn_metadata_ptr->block_tables.has_value() || !attn_metadata_ptr->slot_mapping.has_value() || !attn_metadata_ptr->total_sequence_lengths.has_value() || !attn_metadata_ptr->input_offsets.has_value()) {
            throw std::runtime_error("DeepseekV4Attention::forward requires paged attention metadata");
        }
        if (forward_context.deepseek_v4_kv_cache_vec.size() <= layer_idx_) {
            throw std::runtime_error("DeepseekV4Attention::forward requires DeepSeek-V4 KV cache allocation");
        }
        layer_cache_ptr = &forward_context.deepseek_v4_kv_cache_vec[layer_idx_];
        if (!layer_cache_ptr->swa_cache_raw) {
            throw std::runtime_error("DeepseekV4Attention::forward found an incomplete DeepSeek-V4 SWA cache");
        }
    }
    auto &attn_metadata = *attn_metadata_ptr;
    auto &dsv4_metadata = *dsv4_metadata_ptr;
    auto &layer_cache = *layer_cache_ptr;

    {
        profile::ScopedTimer timer(profile::Event::AttentionRope, seq_len);
        auto q_rope = q->narrow({{2, head_dim_ - qk_rope_head_dim_, qk_rope_head_dim_}});
        auto kv_rope = kv->narrow({{1, head_dim_ - qk_rope_head_dim_, qk_rope_head_dim_}})->unsqueeze(1);
        apply_rope_(pos_ids, q_rope, kv_rope, false);
    }

    const auto device = hidden_states->device();
    if (!dsv4_metadata.swa_indices || !dsv4_metadata.swa_topk_lengths) {
        throw std::runtime_error("DeepseekV4Attention::forward requires DeepSeek-V4 SWA attention metadata");
    }
    auto swa_indices = dsv4_metadata.swa_indices;
    auto swa_topk_lengths = dsv4_metadata.swa_topk_lengths;
    auto cache_slots = dsv4_metadata.raw_out_loc ? dsv4_metadata.raw_out_loc : attn_metadata.slot_mapping.value();

    {
        profile::ScopedTimer timer(profile::Event::AttentionSWAStore, seq_len);
        infinicore::op::deepseek_v4_store_flashmla_raw_cache_(kv,
                                                              layer_cache.swa_cache_raw,
                                                              cache_slots,
                                                              static_cast<int>(kDsv4SwaBlockSize));
    }

    std::optional<infinicore::Tensor> extra_raw_cache = std::nullopt;
    std::optional<infinicore::Tensor> extra_indices = std::nullopt;
    std::optional<infinicore::Tensor> extra_topk_lengths = std::nullopt;
    int extra_page_size = 0;
    if (compress_ratio_ == 4) {
        if (!compressor_ || !layer_cache.c4_cache_raw || !layer_cache.compressor_state || !dsv4_metadata.c4_out_loc || !dsv4_metadata.c4_positions || !dsv4_metadata.c4_topk_lengths_raw || !dsv4_metadata.c4_sparse_indices || !dsv4_metadata.c4_sparse_topk_lengths || !dsv4_metadata.c4_compress_write_loc || !dsv4_metadata.c4_compress_extra_loc || !dsv4_metadata.page_table) {
            throw std::runtime_error("DeepseekV4Attention::forward requires C4 compressor/cache/state/metadata for compressed layer");
        }
        auto c4_out_loc = dsv4_metadata.c4_out_loc;
        auto c4_positions = dsv4_metadata.c4_positions;
        auto c4_write_loc = dsv4_metadata.c4_compress_write_loc;
        auto c4_extra_loc = dsv4_metadata.c4_compress_extra_loc;

        {
            profile::ScopedTimer timer(profile::Event::AttentionC4Compress, seq_len);
            auto c4_kv_score = compressor_->forward_kv_score(hidden_states);
            if (c4_kv_score->ndim() != 2 || c4_kv_score->size(1) < 2 * head_dim_) {
                throw std::runtime_error("DeepseekV4Attention::forward C4 compressor output shape mismatch");
            }
            auto c4_kv = infinicore::op::deepseek_v4_c4_compress_stateful(c4_kv_score,
                                                                          compressor_->ape(),
                                                                          layer_cache.compressor_state,
                                                                          c4_write_loc,
                                                                          c4_extra_loc,
                                                                          pos_ids);
            infinicore::op::deepseek_v4_compress_fused_norm_rope_(c4_kv,
                                                                  compressor_->norm_weight(),
                                                                  compressor_->norm_eps(),
                                                                  rope_freqs_cis_,
                                                                  c4_positions);
            infinicore::op::deepseek_v4_store_flashmla_raw_cache_(c4_kv,
                                                                  layer_cache.c4_cache_raw,
                                                                  c4_out_loc,
                                                                  static_cast<int>(kDsv4C4PageSize));
        }

        if (!indexer_ || !layer_cache.c4_indexer_cache_raw || !layer_cache.indexer_compressor_state) {
            throw std::runtime_error("DeepseekV4Attention::forward requires C4 indexer/cache/state for compressed C4 layer");
        }
        infinicore::Tensor indexer_q;
        infinicore::Tensor indexer_weights;
        {
            profile::ScopedTimer timer(profile::Event::AttentionC4IndexerCompress, seq_len);
            auto indexer_kv_score = indexer_->forward_kv_score(hidden_states);
            if (indexer_kv_score->ndim() != 2 || indexer_kv_score->size(1) != 4 * index_head_dim_) {
                throw std::runtime_error("DeepseekV4Attention::forward C4 indexer compressor output shape mismatch");
            }
            auto indexer_kv = infinicore::op::deepseek_v4_c4_compress_stateful(indexer_kv_score,
                                                                               indexer_->ape(),
                                                                               layer_cache.indexer_compressor_state,
                                                                               c4_write_loc,
                                                                               c4_extra_loc,
                                                                               pos_ids);
            infinicore::op::deepseek_v4_compress_fused_norm_rope_(indexer_kv,
                                                                  indexer_->norm_weight(),
                                                                  indexer_->norm_eps(),
                                                                  rope_freqs_cis_,
                                                                  c4_positions);
            infinicore::op::deepseek_v4_indexer_rotate_(indexer_kv, true);
            infinicore::op::deepseek_v4_store_indexer_raw_cache_(indexer_kv,
                                                                 layer_cache.c4_indexer_cache_raw,
                                                                 c4_out_loc,
                                                                 static_cast<int>(kDsv4C4PageSize));
        }
        {
            profile::ScopedTimer timer(profile::Event::AttentionC4IndexerQuery, seq_len);
            indexer_q = indexer_->compute_q(q_lora, seq_len);
            auto indexer_q_rope = indexer_q->narrow({{2, index_head_dim_ - qk_rope_head_dim_, qk_rope_head_dim_}});
            apply_rope_(pos_ids, indexer_q_rope, std::nullopt, false);
            infinicore::op::deepseek_v4_indexer_rotate_(indexer_q, true);
            indexer_weights = indexer_->compute_weights(hidden_states);
        }
        {
            profile::ScopedTimer timer(profile::Event::AttentionC4IndexerSparse, seq_len);
            const auto max_c4_seq_len = static_cast<int>(dsv4_metadata.page_table->size(1) * kDsv4C4PageSize);
            infinicore::op::deepseek_v4_c4_sparse_attn_indexer_no_logits_(indexer_q,
                                                                          indexer_weights,
                                                                          layer_cache.c4_indexer_cache_raw,
                                                                          dsv4_metadata.c4_topk_lengths_raw,
                                                                          dsv4_metadata.page_table,
                                                                          dsv4_metadata.c4_sparse_indices,
                                                                          max_c4_seq_len,
                                                                          static_cast<int>(kDsv4C4PageSize),
                                                                          indexer_->weight_scale(),
                                                                          false);
        }

        extra_raw_cache = layer_cache.c4_cache_raw;
        extra_indices = dsv4_metadata.c4_sparse_indices;
        extra_topk_lengths = dsv4_metadata.c4_sparse_topk_lengths;
        extra_page_size = static_cast<int>(kDsv4C4PageSize);
    }

    if (compress_ratio_ == 128) {
        if (!compressor_ || !layer_cache.c128_cache_raw || !layer_cache.compressor_state || !dsv4_metadata.c128_out_loc || !dsv4_metadata.c128_positions || !dsv4_metadata.c128_page_indices || !dsv4_metadata.c128_topk_lengths_clamp1 || !dsv4_metadata.c128_compress_write_loc) {
            throw std::runtime_error("DeepseekV4Attention::forward requires C128 compressor/cache/state/metadata for compressed layer");
        }
        auto c128_out_loc = dsv4_metadata.c128_out_loc;
        auto c128_positions = dsv4_metadata.c128_positions;
        auto c128_write_loc = dsv4_metadata.c128_compress_write_loc;

        {
            profile::ScopedTimer timer(profile::Event::AttentionC128Compress, seq_len);
            auto c128_kv_score = compressor_->forward_kv_score(hidden_states);
            if (c128_kv_score->ndim() != 2 || c128_kv_score->size(1) != 2 * head_dim_) {
                throw std::runtime_error("DeepseekV4Attention::forward C128 compressor output shape mismatch");
            }
            auto c128_kv = infinicore::op::deepseek_v4_c128_compress_stateful(c128_kv_score,
                                                                              compressor_->ape(),
                                                                              layer_cache.compressor_state,
                                                                              c128_write_loc,
                                                                              pos_ids);
            infinicore::op::deepseek_v4_compress_fused_norm_rope_(c128_kv,
                                                                  compressor_->norm_weight(),
                                                                  compressor_->norm_eps(),
                                                                  rope_freqs_cis_,
                                                                  c128_positions);
            infinicore::op::deepseek_v4_store_flashmla_raw_cache_(c128_kv,
                                                                  layer_cache.c128_cache_raw,
                                                                  c128_out_loc,
                                                                  static_cast<int>(kDsv4C128PageSize));
        }

        extra_raw_cache = layer_cache.c128_cache_raw;
        extra_indices = dsv4_metadata.c128_page_indices;
        extra_topk_lengths = dsv4_metadata.c128_topk_lengths_clamp1;
        extra_page_size = static_cast<int>(kDsv4C128PageSize);
    }

    auto q_for_flash = q;
    infinicore::Tensor attn_out;
    {
        profile::ScopedTimer timer(profile::Event::AttentionWorkspace, seq_len);
        attn_out = attn_out_workspace(seq_len, hidden_states->dtype(), device);
    }
    auto attn_sink_for_flash = attn_sink_;
    if (tp_size_ > 1) {
        const size_t local_head_start = tp_rank_ * num_local_attention_heads_;
        attn_sink_for_flash = attn_sink_->narrow({{0, local_head_start, num_local_attention_heads_}});
    }

    infinicore::Tensor *flashmla_tile_scheduler_metadata = nullptr;
    infinicore::Tensor *flashmla_num_splits = nullptr;
    std::optional<infinicore::Tensor> flashmla_tile_scheduler_metadata_opt = std::nullopt;
    std::optional<infinicore::Tensor> flashmla_num_splits_opt = std::nullopt;
    {
        profile::ScopedTimer timer(profile::Event::AttentionFlashMLASchedule, seq_len);
        auto flashmla_schedule_tensors = dsv4_metadata.flashmla_schedule_for_compress_ratio(compress_ratio_);
        flashmla_tile_scheduler_metadata = flashmla_schedule_tensors.tile_scheduler_metadata;
        flashmla_num_splits = flashmla_schedule_tensors.num_splits;
        if (*flashmla_tile_scheduler_metadata) {
            flashmla_tile_scheduler_metadata_opt = *flashmla_tile_scheduler_metadata;
        }
        if (*flashmla_num_splits) {
            flashmla_num_splits_opt = *flashmla_num_splits;
        }
    }

    infinicore::op::DeepseekV4FlashMLASparseAttentionSchedule flashmla_schedule;
    {
        profile::ScopedTimer timer(profile::Event::AttentionFlashMLA, seq_len);
        flashmla_schedule = infinicore::op::deepseek_v4_flashmla_sparse_attention_with_metadata_(q_for_flash,
                                                                                                 layer_cache.swa_cache_raw,
                                                                                                 swa_indices,
                                                                                                 swa_topk_lengths,
                                                                                                 attn_sink_for_flash,
                                                                                                 attn_out,
                                                                                                 flashmla_tile_scheduler_metadata_opt,
                                                                                                 flashmla_num_splits_opt,
                                                                                                 static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_))),
                                                                                                 static_cast<int>(kDsv4SwaBlockSize),
                                                                                                 static_cast<int>(head_dim_),
                                                                                                 extra_raw_cache,
                                                                                                 extra_indices,
                                                                                                 extra_topk_lengths,
                                                                                                 extra_page_size);
    }
    {
        profile::ScopedTimer timer(profile::Event::AttentionFlashMLASchedule, seq_len);
        if (!*flashmla_tile_scheduler_metadata && flashmla_schedule.tile_scheduler_metadata) {
            *flashmla_tile_scheduler_metadata = flashmla_schedule.tile_scheduler_metadata;
        }
        if (!*flashmla_num_splits && flashmla_schedule.num_splits) {
            *flashmla_num_splits = flashmla_schedule.num_splits;
        }
    }

    {
        profile::ScopedTimer timer(profile::Event::AttentionOutRope, seq_len);
        auto out_rope = attn_out->narrow({{2, head_dim_ - qk_rope_head_dim_, qk_rope_head_dim_}});
        apply_rope_(pos_ids, out_rope, std::nullopt, true);
    }

    auto wo_a_in = attn_out->view({seq_len, num_local_attention_heads_ * head_dim_ / num_local_groups_});
    infinicore::Tensor wo_a_out;
    {
        profile::ScopedTimer timer(profile::Event::AttentionWoA, seq_len);
        wo_a_out = wo_a_->forward(wo_a_in);
    }
    {
        profile::ScopedTimer timer(profile::Event::AttentionWoB, seq_len);
        return wo_b_->forward(wo_a_out);
    }
}

} // namespace infinilm::models::deepseek_v4
