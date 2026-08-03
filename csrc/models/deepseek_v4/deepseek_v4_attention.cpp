#include "deepseek_v4_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "deepseek_v4_profile.hpp"
#include "deepseek_v4_rope.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_cache.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"
#include "infinicore/ops/deepseek_v4_fused_rope.hpp"
#include "infinicore/ops/deepseek_v4_rmsnorm_self.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::deepseek_v4 {

thread_local DeepseekV4AttentionScratch DeepseekV4Attention::attention_scratch_;
thread_local DeepseekV4MLAScratch DeepseekV4Attention::mla_scratch_;

DeepseekV4Attention::DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         size_t layer_idx,
                                         const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto dtype = model_config->get_dtype();
    dtype_ = dtype;
    device_ = device;
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
    q_lora_rank_ = model_config->get<size_t>("q_lora_rank"); // 1024
    o_lora_rank_ = model_config->get<size_t>("o_lora_rank");
    qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim"); // 64
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
        compressor_ = this->register_module<DeepseekV4Compressor>("compressor", model_config, head_dim_, compress_ratio_, device);
        INFINICORE_NN_MODULE_INIT(indexer, model_config, device);
    } else if (compress_ratio_ == 128) {
        compressor_ = this->register_module<DeepseekV4Compressor>("compressor", model_config, head_dim_, compress_ratio_, device);
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

void copy_flashmla_schedule_tensor(infinicore::Tensor &dst,
                                   const infinicore::Tensor &src) {
    if (!src) {
        throw std::runtime_error("DeepseekV4Attention: empty FlashMLA schedule tensor");
    }
    if (src->dtype() != infinicore::DataType::I32) {
        throw std::runtime_error("DeepseekV4Attention: FlashMLA schedule tensor must be int32");
    }
    if (!src->is_contiguous()) {
        throw std::runtime_error("DeepseekV4Attention: FlashMLA schedule tensor must be contiguous");
    }

    if (!dst || dst->shape() != src->shape() || dst->dtype() != src->dtype() || dst->device() != src->device()) {
        dst = infinicore::Tensor::empty(src->shape(), src->dtype(), src->device());
    }

    const size_t bytes = src->numel() * sizeof(std::int32_t);
    infinicore::context::memcpyD2D(dst->data(), src->data(), bytes, false);
}

} // namespace

// 复用同一线程内 attention 的 attn_out 输出 buffer；shape/dtype/device 不匹配时重新分配。
infinicore::Tensor DeepseekV4Attention::prepare_attn_out_workspace(size_t seq_len,
                                                                   infinicore::DataType dtype,
                                                                   const infinicore::Device &device) const {
    const infinicore::Shape expected_shape{seq_len, num_local_attention_heads_, head_dim_};
    return attention_scratch_.get_attn_out(expected_shape, dtype, device);
}

void DeepseekV4Attention::cache_flashmla_schedule_metadata(
    infinilm::global_state::FlashMLASchedMeta &flashmla_metadata,
    const infinicore::op::DeepseekV4FlashMLASparseAttentionSchedule &flashmla_schedule) const {
    if (!flashmla_metadata.tile_scheduler_metadata && flashmla_schedule.tile_scheduler_metadata) {
        copy_flashmla_schedule_tensor(flashmla_metadata.tile_scheduler_metadata,
                                      flashmla_schedule.tile_scheduler_metadata);
    }
    if (!flashmla_metadata.num_splits && flashmla_schedule.num_splits) {
        copy_flashmla_schedule_tensor(flashmla_metadata.num_splits,
                                      flashmla_schedule.num_splits);
    }
    flashmla_metadata.have_initialized = flashmla_metadata.tile_scheduler_metadata && flashmla_metadata.num_splits;
}

void DeepseekV4Attention::compute_sparse_attention(
    infinicore::Tensor attn_out,
    const infinicore::Tensor &q, // [tokens, num_attention_heads , head_dim]
    size_t seq_len,
    const infinicore::Device &device,
    const infinicore::Tensor &swa_cache_raw,
    const infinicore::Tensor &swa_indices,
    const infinicore::Tensor &swa_topk_lengths,
    std::optional<infinicore::Tensor> extra_raw_cache,
    std::optional<infinicore::Tensor> extra_indices,
    std::optional<infinicore::Tensor> extra_topk_lengths,
    int extra_page_size,
    infinilm::global_state::DSV4AttnMetadata &dsv4_metadata) const {
    infinicore::op::DeepseekV4FlashMLASparseAttentionSchedule flashmla_schedule;
    auto q_for_flash = q; // [tokens, num_attention_heads , head_dim]

    std::optional<infinicore::Tensor> flashmla_tile_scheduler_metadata_opt;
    std::optional<infinicore::Tensor> flashmla_num_splits_opt;
    infinilm::global_state::FlashMLASchedMeta *flashmla_metadata = nullptr;
    {
        profile::ScopedTimer timer(profile::Event::AttentionFlashMLASchedule, seq_len);
        flashmla_metadata = &dsv4_metadata.get_flashmla_metadata(compress_ratio_);
        if (flashmla_metadata->tile_scheduler_metadata) {
            flashmla_tile_scheduler_metadata_opt = flashmla_metadata->tile_scheduler_metadata;
        }
        if (flashmla_metadata->num_splits) {
            flashmla_num_splits_opt = flashmla_metadata->num_splits;
        }
    }
    const bool has_flashmla_schedule = flashmla_metadata->have_initialized
                                    && flashmla_tile_scheduler_metadata_opt.has_value() && flashmla_tile_scheduler_metadata_opt.value()
                                    && flashmla_num_splits_opt.has_value() && flashmla_num_splits_opt.value();

    if (has_flashmla_schedule) {
        profile::ScopedTimer timer(profile::Event::AttentionFlashMLA, seq_len);
        const auto num_sm_parts = flashmla_tile_scheduler_metadata_opt.value()->size(0);
        infinicore::Tensor flashmla_lse_workspace;
        infinicore::Tensor flashmla_lse_accum_workspace;
        infinicore::Tensor flashmla_o_accum_workspace;
        {
            profile::ScopedTimer sub_timer(profile::Event::AttentionFlashMLAWorkspace, seq_len);
            flashmla_lse_workspace = mla_scratch_.get_lse(
                {seq_len, num_local_attention_heads_},
                infinicore::DataType::F32,
                device);
            flashmla_lse_accum_workspace = mla_scratch_.get_lse_accum(
                {seq_len + num_sm_parts, num_local_attention_heads_},
                infinicore::DataType::F32,
                device);
            flashmla_o_accum_workspace = mla_scratch_.get_o_accum(
                {seq_len + num_sm_parts, num_local_attention_heads_, head_dim_},
                infinicore::DataType::F32,
                device);
        }
        profile::ScopedTimer sub_timer(profile::Event::AttentionFlashMLAOutWorkspaceCall, seq_len);
        infinicore::op::deepseek_v4_flashmla_sparse_attention_out_workspace_(q_for_flash,
                                                                             swa_cache_raw,
                                                                             swa_indices,
                                                                             swa_topk_lengths,
                                                                             attn_sink_for_flash_,
                                                                             attn_out,
                                                                             flashmla_lse_workspace,
                                                                             flashmla_lse_accum_workspace,
                                                                             flashmla_o_accum_workspace,
                                                                             flashmla_tile_scheduler_metadata_opt.value(),
                                                                             flashmla_num_splits_opt.value(),
                                                                             flashmla_softmax_scale_,
                                                                             static_cast<int>(kDsv4SwaBlockSize),
                                                                             static_cast<int>(head_dim_),
                                                                             extra_raw_cache,
                                                                             extra_indices,
                                                                             extra_topk_lengths,
                                                                             extra_page_size);
    } else if (!has_flashmla_schedule) {
        profile::ScopedTimer timer(profile::Event::AttentionFlashMLA, seq_len);
        {
            profile::ScopedTimer sub_timer(profile::Event::AttentionFlashMLAWithMetadataCall, seq_len);
            flashmla_schedule = infinicore::op::deepseek_v4_flashmla_sparse_attention_with_metadata_(q_for_flash,
                                                                                                     swa_cache_raw,
                                                                                                     swa_indices,
                                                                                                     swa_topk_lengths,
                                                                                                     attn_sink_for_flash_,
                                                                                                     attn_out,
                                                                                                     flashmla_tile_scheduler_metadata_opt,
                                                                                                     flashmla_num_splits_opt,
                                                                                                     flashmla_softmax_scale_,
                                                                                                     static_cast<int>(kDsv4SwaBlockSize),
                                                                                                     static_cast<int>(head_dim_),
                                                                                                     extra_raw_cache,
                                                                                                     extra_indices,
                                                                                                     extra_topk_lengths,
                                                                                                     extra_page_size);
        }
        {
            profile::ScopedTimer sub_timer(profile::Event::AttentionFlashMLACacheMetadata, seq_len);
            cache_flashmla_schedule_metadata(*flashmla_metadata, flashmla_schedule);
        }
    }
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
    const auto &dsv4_metadata = forward_context.dsv4_attn_metadata;
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
        ;
    } else if (compress_ratio_ == 4) {
        if (!compressor_ || !indexer_ || !layer_cache.c4_cache_raw || !layer_cache.compressor_state || !dsv4_metadata.c4_out_loc || !dsv4_metadata.c4_positions || !dsv4_metadata.c4_topk_lengths_raw || !dsv4_metadata.c4_sparse_topk_lengths || !dsv4_metadata.c4_compress_write_loc || !dsv4_metadata.c4_compress_extra_loc || !dsv4_metadata.page_table) {
            throw std::runtime_error("DeepseekV4Attention::forward requires C4 compressor/cache/state/metadata for compressed layer");
        }
        if (!layer_cache.c4_indexer_cache_raw || !layer_cache.indexer_compressor_state) {
            throw std::runtime_error("DeepseekV4Attention::forward requires C4 indexer/cache/state for compressed C4 layer");
        }
    } else if (compress_ratio_ == 128) {
        if (!compressor_ || !layer_cache.c128_cache_raw || !layer_cache.compressor_state || !dsv4_metadata.c128_out_loc || !dsv4_metadata.c128_positions || !dsv4_metadata.c128_page_indices || !dsv4_metadata.c128_topk_lengths_clamp1 || !dsv4_metadata.c128_compress_write_loc) {
            throw std::runtime_error("DeepseekV4Attention::forward requires C128 compressor/cache/state/metadata for compressed layer");
        }

    } else {
        throw std::runtime_error("DeepseekV4Attention::forward found unsupported compress_ratio");
    }
}

infinicore::Tensor DeepseekV4Attention::forward(const infinicore::Tensor &positions,
                                                const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape(); // [tokens, hidden]
    if (shape.size() != 2) {
        throw std::runtime_error("DeepseekV4Attention::forward expects hidden_states [tokens, hidden]");
    }
    const size_t seq_len = shape[0];
    profile::ScopedTimer forward_timer(profile::Event::AttentionForward, seq_len);

    const auto pos_shape = positions->shape(); // [tokens]
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
    auto &dsv4_metadata = forward_context.dsv4_attn_metadata;
    auto &layer_cache = forward_context.deepseek_v4_kv_cache_vec[layer_idx_];

    infinicore::Tensor q_lora;
    infinicore::Tensor q;
    {
        profile::ScopedTimer timer(profile::Event::AttentionQProjection, seq_len);
        auto x0 = hidden_states;
        {
            profile::ScopedTimer sub_timer(profile::Event::AttentionQProjA, seq_len);
            q_lora = wq_a_->forward(x0); // [tokens, hidden] => [tokens, q_lora_rank]
        }
        {
            profile::ScopedTimer sub_timer(profile::Event::AttentionQNorm, seq_len);
            q_lora = q_norm_->forward(q_lora);
        }
        auto q_mut = q_lora;
        {
            profile::ScopedTimer sub_timer(profile::Event::AttentionQProjB, seq_len);

            // [tokens, q_lora_rank] => [tokens, num_attention_heads * head_dim]
            q = wq_b_->forward(q_mut)->view({seq_len, num_local_attention_heads_, head_dim_});
        }
        {
            profile::ScopedTimer sub_timer(profile::Event::AttentionQRmsNormSelf, seq_len);
            q = infinicore::op::deepseek_v4_rmsnorm_self(q, static_cast<float>(rms_norm_eps_));
        }
    }

    infinicore::Tensor kv;
    {
        profile::ScopedTimer timer(profile::Event::AttentionKVProjection, seq_len);
        auto x1 = hidden_states;
        {
            profile::ScopedTimer sub_timer(profile::Event::AttentionKVProj, seq_len);
            kv = wkv_->forward(x1); // [tokens, hidden] => [tokens, head_dim]
        }
        {
            profile::ScopedTimer sub_timer(profile::Event::AttentionKVNorm, seq_len);
            kv = kv_norm_->forward(kv);
        }
    }

    {
        profile::ScopedTimer timer(profile::Event::AttentionRope, seq_len);
        // q:  [tokens, num_local_attention_heads, head_dim]
        // kv: [tokens, head_dim]
        // qk_rope_head_dim是64；head_dim是512.

        // q_rope 是 [tokens, num_local_attention_heads, qk_rope_head_dim]
        // kv_rope 是 [tokens, 1, qk_rope_head_dim]
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
        const auto &c4_out_loc = dsv4_metadata.c4_out_loc;
        const auto &c4_positions = dsv4_metadata.c4_positions;
        const auto &c4_write_loc = dsv4_metadata.c4_compress_write_loc;
        const auto &c4_extra_loc = dsv4_metadata.c4_compress_extra_loc;
        compressor_->forward(hidden_states,
                             pos_ids,
                             seq_len,
                             rope_freqs_cis_,
                             layer_cache.compressor_state,
                             layer_cache.c4_cache_raw,
                             c4_out_loc,
                             c4_positions,
                             c4_write_loc,
                             c4_extra_loc);
        infinicore::Tensor c4_sparse_indices;
        {
            profile::ScopedTimer timer(profile::Event::AttentionC4SparseAlloc, seq_len);
            c4_sparse_indices = infinicore::Tensor::empty({seq_len, kDsv4C4Topk},
                                                          infinicore::DataType::I32,
                                                          hidden_states->device());
        }
        indexer_->forward(hidden_states,
                          q_lora,
                          pos_ids,
                          seq_len,
                          rope_freqs_cis_,
                          qk_rope_head_dim_,
                          layer_cache.indexer_compressor_state,
                          layer_cache.c4_indexer_cache_raw,
                          c4_out_loc,
                          c4_positions,
                          c4_write_loc,
                          c4_extra_loc,
                          dsv4_metadata.c4_topk_lengths_raw,
                          dsv4_metadata.page_table,
                          c4_sparse_indices);

        extra_raw_cache = layer_cache.c4_cache_raw;
        extra_indices = c4_sparse_indices;
        extra_topk_lengths = dsv4_metadata.c4_sparse_topk_lengths;
        extra_page_size = static_cast<int>(kDsv4C4PageSize);

    } else if (compress_ratio_ == 128) {
        const auto &c128_out_loc = dsv4_metadata.c128_out_loc;
        const auto &c128_positions = dsv4_metadata.c128_positions;
        const auto &c128_write_loc = dsv4_metadata.c128_compress_write_loc;

        compressor_->forward(hidden_states,
                             pos_ids,
                             seq_len,
                             rope_freqs_cis_,
                             layer_cache.compressor_state,
                             layer_cache.c128_cache_raw,
                             c128_out_loc,
                             c128_positions,
                             c128_write_loc,
                             std::nullopt);

        extra_raw_cache = layer_cache.c128_cache_raw;
        extra_indices = dsv4_metadata.c128_page_indices;
        extra_topk_lengths = dsv4_metadata.c128_topk_lengths_clamp1;
        extra_page_size = static_cast<int>(kDsv4C128PageSize);
    }

    // swa_indices 是 SWA 稀疏注意力读取的 cache page 索引；
    // swa_topk_lengths 是每个 token 实际使用的 SWA page 数。
    // flashmla_schedule_cache 在一次 forward 内缓存 FlashMLA 调度 metadata，
    // 供同类 attention 的不同 decoder layer 复用。
    infinicore::Tensor attn_out;
    {
        profile::ScopedTimer timer(profile::Event::AttentionWorkspace, seq_len);
        attn_out = prepare_attn_out_workspace(seq_len, hidden_states->dtype(), hidden_states->device());
    }
    compute_sparse_attention(attn_out,
                             q,
                             seq_len,
                             hidden_states->device(),
                             layer_cache.swa_cache_raw,
                             dsv4_metadata.swa_indices,
                             dsv4_metadata.swa_topk_lengths,
                             extra_raw_cache,
                             extra_indices,
                             extra_topk_lengths,
                             extra_page_size,
                             dsv4_metadata);

    {
        profile::ScopedTimer timer(profile::Event::AttentionOutRope, seq_len);
        // attn_out是 [seq_len, num_local_attention_heads_, head_dim_]
        auto out_rope = attn_out->narrow({{2, head_dim_ - qk_rope_head_dim_, qk_rope_head_dim_}});
        apply_rope_(pos_ids, out_rope, std::nullopt, true);
    }

    auto wo_a_in = attn_out->view({seq_len, num_local_attention_heads_ * head_dim_ / num_local_groups_});
    infinicore::Tensor wo_a_out;
    {
        profile::ScopedTimer timer(profile::Event::AttentionWoA, seq_len);
        //  [seq_len, num_local_attention_heads_ * head_dim_ / o_groups_]
        // =>  [seq_len, o_groups_ * o_lora_rank_ ]
        wo_a_out = wo_a_->forward(wo_a_in);
    }
    {

        // [seq_len, o_groups_ * o_lora_rank_ ]
        //  => [seq_len, hidden_size_ ]
        profile::ScopedTimer timer(profile::Event::AttentionWoB, seq_len);
        return wo_b_->forward(wo_a_out);
    }
}

} // namespace infinilm::models::deepseek_v4
