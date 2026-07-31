#include "deepseek_v4_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "deepseek_v4_profile.hpp"
#include "deepseek_v4_rope.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_cache.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"
#include "infinicore/ops/deepseek_v4_fused_rope.hpp"
#include "infinicore/ops/deepseek_v4_rmsnorm_self.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {

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
    flashmla_softmax_scale_ = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

    INFINICORE_NN_PARAMETER_INIT(attn_sink, ({num_attention_heads_}, infinicore::DataType::F32, device));
    attn_sink_for_flash_ = attn_sink_;
    if (tp_size_ > 1) {
        const size_t local_head_start = tp_rank_ * num_local_attention_heads_;
        attn_sink_for_flash_ = attn_sink_->narrow({{0, local_head_start, num_local_attention_heads_}});
    }

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
    if (compress_ratio_ == 4) {
        csa_compressor_ = this->register_module<DeepseekV4CSACompressor>("compressor", model_config, head_dim_, device);
        INFINICORE_NN_MODULE_INIT(indexer, model_config, device);
    } else if (compress_ratio_ == 128) {
        hca_compressor_ = this->register_module<DeepseekV4HCACompressor>("compressor", model_config, head_dim_, device);
    } else if (compress_ratio_ != 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4Attention: unsupported compress_ratio");
    }

    rope_freqs_cis_ = build_deepseek_v4_rope_freqs_cis(qk_rope_head_dim_,
                                                       max_position_embeddings_,
                                                       compress_ratio_ != 0,
                                                       rope_theta_,
                                                       compress_rope_theta_,
                                                       rope_factor_,
                                                       rope_beta_fast_,
                                                       rope_beta_slow_,
                                                       rope_original_seq_len_,
                                                       device);
}

namespace {

constexpr size_t kDsv4SwaBlockSize = 256;
constexpr size_t kDsv4C4PageSize = 64;
constexpr size_t kDsv4C4Topk = 512;
constexpr size_t kDsv4C128PageSize = 2;

} // namespace

infinicore::Tensor DeepseekV4Attention::prepare_attn_out_workspace(size_t seq_len,
                                                                   infinicore::DataType dtype,
                                                                   const infinicore::Device &device) const {
    const infinicore::Shape expected_shape{seq_len, num_local_attention_heads_, head_dim_};
    if (!attn_out_workspace_ || attn_out_workspace_->shape() != expected_shape || attn_out_workspace_->dtype() != dtype || attn_out_workspace_->device() != device) {
        attn_out_workspace_ = infinicore::Tensor::empty(expected_shape, dtype, device);
    }
    return attn_out_workspace_;
}

infinicore::Tensor DeepseekV4Attention::flashmla_workspace(std::vector<infinicore::Tensor> &cache,
                                                           const infinicore::Shape &shape,
                                                           infinicore::DataType dtype,
                                                           const infinicore::Device &device) const {
    for (auto &workspace : cache) {
        if (workspace && workspace->shape() == shape && workspace->dtype() == dtype && workspace->device() == device) {
            return workspace;
        }
    }
    cache.push_back(infinicore::Tensor::empty(shape, dtype, device));
    return cache.back();
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

void DeepseekV4Attention::validate_forward_metadata_and_cache(
    const infinilm::global_state::ForwardContext &forward_context) const {
    // Check all metadata/cache prerequisites that are known before launching attention work.
    const auto &attn_metadata = forward_context.attn_metadata;
    const auto &dsv4_metadata = forward_context.deepseek_v4_attention_metadata;
    if (!attn_metadata.block_tables.has_value() || !attn_metadata.slot_mapping.has_value() || !attn_metadata.total_sequence_lengths.has_value() || !attn_metadata.input_offsets.has_value()) {
        throw std::runtime_error("DeepseekV4Attention::forward requires paged attention metadata");
    }
    if (forward_context.deepseek_v4_kv_cache_vec.size() <= layer_idx_) {
        throw std::runtime_error("DeepseekV4Attention::forward requires DeepSeek-V4 KV cache allocation");
    }

    const auto &layer_cache = forward_context.deepseek_v4_kv_cache_vec[layer_idx_];
    if (!layer_cache.swa_cache_raw) {
        throw std::runtime_error("DeepseekV4Attention::forward found an incomplete DeepSeek-V4 SWA cache");
    }
    if (!dsv4_metadata.swa_indices || !dsv4_metadata.swa_topk_lengths) {
        throw std::runtime_error("DeepseekV4Attention::forward requires DeepSeek-V4 SWA attention metadata");
    }

    if (compress_ratio_ == 0) {
        return;
    }
    if (compress_ratio_ == 4) {
        if (!csa_compressor_ || !indexer_ || !layer_cache.c4_cache_raw || !layer_cache.compressor_state || !dsv4_metadata.c4_out_loc || !dsv4_metadata.c4_positions || !dsv4_metadata.c4_topk_lengths_raw || !dsv4_metadata.c4_sparse_topk_lengths || !dsv4_metadata.c4_compress_write_loc || !dsv4_metadata.c4_compress_extra_loc || !dsv4_metadata.page_table) {
            throw std::runtime_error("DeepseekV4Attention::forward requires C4 compressor/cache/state/metadata for compressed layer");
        }
        if (!layer_cache.c4_indexer_cache_raw || !layer_cache.indexer_compressor_state) {
            throw std::runtime_error("DeepseekV4Attention::forward requires C4 indexer/cache/state for compressed C4 layer");
        }
        return;
    }
    if (compress_ratio_ == 128) {
        if (!hca_compressor_ || !layer_cache.c128_cache_raw || !layer_cache.compressor_state || !dsv4_metadata.c128_out_loc || !dsv4_metadata.c128_positions || !dsv4_metadata.c128_page_indices || !dsv4_metadata.c128_topk_lengths_clamp1 || !dsv4_metadata.c128_compress_write_loc) {
            throw std::runtime_error("DeepseekV4Attention::forward requires C128 compressor/cache/state/metadata for compressed layer");
        }
        return;
    }

    throw std::runtime_error("DeepseekV4Attention::forward found unsupported compress_ratio");
}

infinicore::Tensor DeepseekV4Attention::forward(const infinicore::Tensor &positions,
                                                const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape();
    if (shape.size() != 2) {
        throw std::runtime_error("DeepseekV4Attention::forward expects hidden_states [tokens, hidden]");
    }
    const size_t seq_len = shape[0];
    profile::ScopedTimer forward_timer(profile::Event::AttentionForward, seq_len);

    const auto pos_shape = positions->shape();
    if (pos_shape.size() != 1 || pos_shape[0] != seq_len) {
        throw std::runtime_error("DeepseekV4Attention::forward expects positions [tokens]");
    }
    auto pos_ids = positions;

    auto &forward_context = infinilm::global_state::get_forward_context();
    {
        profile::ScopedTimer timer(profile::Event::AttentionMetadata, seq_len);
        validate_forward_metadata_and_cache(forward_context);
    }
    auto &attn_metadata = forward_context.attn_metadata;
    auto &dsv4_metadata = forward_context.deepseek_v4_attention_metadata;
    auto &flashmla_schedule_cache = forward_context.deepseek_v4_flashmla_schedule_cache;
    auto &layer_cache = forward_context.deepseek_v4_kv_cache_vec[layer_idx_];

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

    {
        profile::ScopedTimer timer(profile::Event::AttentionRope, seq_len);
        auto q_rope = q->narrow({{2, head_dim_ - qk_rope_head_dim_, qk_rope_head_dim_}});
        auto kv_rope = kv->narrow({{1, head_dim_ - qk_rope_head_dim_, qk_rope_head_dim_}})->unsqueeze(1);
        apply_rope_(pos_ids, q_rope, kv_rope, false);
    }

    auto cache_slots = dsv4_metadata.raw_out_loc ? dsv4_metadata.raw_out_loc : attn_metadata.slot_mapping.value();

    {
        profile::ScopedTimer timer(profile::Event::AttentionSWAStore, seq_len);
        // 该注释不要被删除：swa_cache_raw 存储的是普通 KV 的 FlashMLA raw/FP8 page layout。
        // "SWA" 表示这份 cache 服务于 Sliding Window Attention；滑窗语义由后续读取时的
        // swa_indices 和 swa_topk_lengths 选择最近窗口体现，不改变这里写入的 KV 内容。
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
        csa_compressor_->forward(hidden_states,
                                 pos_ids,
                                 seq_len,
                                 rope_freqs_cis_,
                                 layer_cache,
                                 dsv4_metadata);
        auto c4_sparse_indices = infinicore::Tensor::empty({seq_len, kDsv4C4Topk},
                                                           infinicore::DataType::I32,
                                                           hidden_states->device());
        indexer_->forward(hidden_states,
                          q_lora,
                          pos_ids,
                          seq_len,
                          rope_freqs_cis_,
                          qk_rope_head_dim_,
                          layer_cache,
                          dsv4_metadata,
                          c4_sparse_indices);
        extra_raw_cache = layer_cache.c4_cache_raw;
        extra_indices = c4_sparse_indices;
        extra_topk_lengths = dsv4_metadata.c4_sparse_topk_lengths;
        extra_page_size = static_cast<int>(kDsv4C4PageSize);
    } else if (compress_ratio_ == 128) {
        hca_compressor_->forward(hidden_states,
                                 pos_ids,
                                 seq_len,
                                 rope_freqs_cis_,
                                 layer_cache,
                                 dsv4_metadata);
        extra_raw_cache = layer_cache.c128_cache_raw;
        extra_indices = dsv4_metadata.c128_page_indices;
        extra_topk_lengths = dsv4_metadata.c128_topk_lengths_clamp1;
        extra_page_size = static_cast<int>(kDsv4C128PageSize);
    }

    infinicore::Tensor attn_out;
    {
        profile::ScopedTimer timer(profile::Event::AttentionWorkspace, seq_len);
        attn_out = prepare_attn_out_workspace(seq_len, hidden_states->dtype(), hidden_states->device());
    }

    infinicore::op::DeepseekV4FlashMLASparseAttentionSchedule flashmla_schedule;
    const float flashmla_softmax_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));
    const bool has_flashmla_schedule = flashmla_tile_scheduler_metadata_opt.has_value() && flashmla_tile_scheduler_metadata_opt.value() && flashmla_num_splits_opt.has_value() && flashmla_num_splits_opt.value();
    bool used_flashmla_out_workspace = false;
    if (has_flashmla_schedule && !flashmla_out_workspace_disabled_) {
        profile::ScopedTimer timer(profile::Event::AttentionFlashMLA, seq_len);
        const auto num_sm_parts = flashmla_tile_scheduler_metadata_opt.value()->size(0);
        auto lse = flashmla_workspace(flashmla_lse_workspaces_,
                                      {seq_len, num_local_attention_heads_},
                                      infinicore::DataType::F32,
                                      device);
        auto lse_accum = flashmla_workspace(flashmla_lse_accum_workspaces_,
                                            {seq_len + num_sm_parts, num_local_attention_heads_},
                                            infinicore::DataType::F32,
                                            device);
        auto o_accum = flashmla_workspace(flashmla_o_accum_workspaces_,
                                          {seq_len + num_sm_parts, num_local_attention_heads_, head_dim_},
                                          infinicore::DataType::F32,
                                          device);
        try {
            infinicore::op::deepseek_v4_flashmla_sparse_attention_out_workspace_(q_for_flash,
                                                                                 layer_cache.swa_cache_raw,
                                                                                 swa_indices,
                                                                                 swa_topk_lengths,
                                                                                 attn_sink_for_flash,
                                                                                 attn_out,
                                                                                 lse,
                                                                                 lse_accum,
                                                                                 o_accum,
                                                                                 flashmla_tile_scheduler_metadata_opt.value(),
                                                                                 flashmla_num_splits_opt.value(),
                                                                                 flashmla_softmax_scale,
                                                                                 static_cast<int>(kDsv4SwaBlockSize),
                                                                                 static_cast<int>(head_dim_),
                                                                                 extra_raw_cache,
                                                                                 extra_indices,
                                                                                 extra_topk_lengths,
                                                                                 extra_page_size);
            used_flashmla_out_workspace = true;
        } catch (const std::runtime_error &err) {
            const std::string message = err.what();
            if (message.find("out/workspace symbol") == std::string::npos && message.find("flash_mla_sparse_decode_fwd_out_workspace") == std::string::npos) {
                throw;
            }
            flashmla_out_workspace_disabled_ = true;
        }
    }
    if (!used_flashmla_out_workspace) {
        profile::ScopedTimer timer(profile::Event::AttentionFlashMLA, seq_len);
        auto [flashmla_tile_scheduler_metadata_opt, flashmla_num_splits_opt]
            = prepare_flashmla_schedule_metadata(flashmla_schedule_cache);
        flashmla_schedule = infinicore::op::deepseek_v4_flashmla_sparse_attention_with_metadata_(q_for_flash,
                                                                                                 layer_cache.swa_cache_raw,
                                                                                                 swa_indices,
                                                                                                 swa_topk_lengths,
                                                                                                 attn_sink_for_flash_,
                                                                                                 attn_out,
                                                                                                 flashmla_tile_scheduler_metadata_opt,
                                                                                                 flashmla_num_splits_opt,
                                                                                                 flashmla_softmax_scale,
                                                                                                 static_cast<int>(kDsv4SwaBlockSize),
                                                                                                 static_cast<int>(head_dim_),
                                                                                                 extra_raw_cache,
                                                                                                 extra_indices,
                                                                                                 extra_topk_lengths,
                                                                                                 extra_page_size);
        cache_flashmla_schedule_metadata(flashmla_schedule_cache, flashmla_schedule);
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
