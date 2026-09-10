#include "deepseek_v4_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/deepseek_v4/flashmla_cache.hpp"
#include "infinicore/ops/deepseek_v4/fused_rope.hpp"
#include "infinicore/ops/deepseek_v4/rmsnorm_self.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

constexpr size_t kDsv4FlashMlaValueBytesPerToken = 576;
constexpr size_t kDsv4FlashMlaBytesPerToken = 584;

size_t flashmla_raw_cache_page_bytes(size_t page_size) {
    return ((page_size * kDsv4FlashMlaBytesPerToken + kDsv4FlashMlaValueBytesPerToken - 1)
            / kDsv4FlashMlaValueBytesPerToken)
         * kDsv4FlashMlaValueBytesPerToken;
}

size_t infer_flashmla_raw_cache_page_size(const infinicore::Tensor &raw_cache) {
    for (size_t page_size = 1; page_size <= 4096; page_size <<= 1) {
        if (raw_cache->size(1) == flashmla_raw_cache_page_bytes(page_size)) {
            return page_size;
        }
    }
    throw std::runtime_error("DeepseekV4Attention: cannot infer sparse FlashMLA raw cache page size");
}

} // namespace

DeepseekV4Attention::DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         size_t layer_idx,
                                         const infinicore::Device &device)
    : layer_idx_(layer_idx), dtype_(model_config->get_dtype()), device_(device) {
    const auto quantization_method = model_config->get_quantization_method();
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
    tp_size_ = static_cast<size_t>(rank_info.tp_size);

    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    num_attention_heads_ = model_config->get<size_t>("num_attention_heads");
    q_lora_rank_ = model_config->get<size_t>("q_lora_rank");
    o_lora_rank_ = model_config->get<size_t>("o_lora_rank");
    o_groups_ = model_config->get_or<size_t>("o_groups", 1);
    rms_norm_eps_ = model_config->get<double>("rms_norm_eps");
    max_position_embeddings_ = model_config->get_or<size_t>("max_position_embeddings", 131072);
    rope_theta_ = model_config->get_or<double>("rope_theta", 10000.0);
    compress_rope_theta_ = model_config->get_or<double>("compress_rope_theta", 160000.0);
    const auto &config_json = model_config->get_config_json();
    if (config_json.contains("rope_scaling") && config_json["rope_scaling"].is_object()) {
        const auto &rope_scaling = config_json["rope_scaling"];
        rope_factor_ = rope_scaling.value("factor", 1.0);
        rope_beta_fast_ = rope_scaling.value("beta_fast", 32.0);
        rope_beta_slow_ = rope_scaling.value("beta_slow", 1.0);
        rope_original_seq_len_ = rope_scaling.value("original_max_position_embeddings", 0);
    }
    if (config_json.contains("compress_ratios") && config_json["compress_ratios"].is_array()) {
        const auto compress_ratios = config_json["compress_ratios"].get<std::vector<size_t>>();
        compress_ratio_ = layer_idx_ < compress_ratios.size() ? compress_ratios[layer_idx_] : 0;
    }
    if (model_config->get_or<size_t>("num_key_value_heads", 1) != 1) {
        throw std::runtime_error("DeepseekV4Attention: num_key_value_heads must be 1");
    }
    if (tp_size_ == 0 || num_attention_heads_ % tp_size_ != 0 || o_groups_ % tp_size_ != 0) {
        throw std::runtime_error("DeepseekV4Attention: num_attention_heads and o_groups must be divisible by tp_size");
    }
    num_local_attention_heads_ = num_attention_heads_ / tp_size_;
    num_local_groups_ = o_groups_ / tp_size_;
    softmax_scale_ = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));
    sparse_mla_attn_ = std::make_shared<infinilm::layers::mla_attention::SparseFlashMLAImpl>(
        num_local_attention_heads_,
        head_dim_,
        softmax_scale_,
        1,
        head_dim_);

    INFINICORE_NN_PARAMETER_INIT(attn_sink, ({num_attention_heads_}, infinicore::DataType::F32, device));

    auto register_wqkv_a_param = [this](const std::string &name, infinicore::nn::Parameter param) {
        this->register_parameter(name, std::move(param));
    };
    wqkv_a_ = std::make_shared<infinilm::layers::linear::FusedReplicatedLinear>(
        hidden_size_,
        q_lora_rank_,
        head_dim_,
        "wq_a",
        "wkv",
        register_wqkv_a_param,
        quantization_method,
        false,
        dtype_,
        device);
    wq_b_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "wq_b",
        q_lora_rank_,
        num_attention_heads_ * head_dim_,
        quantization_method,
        false,
        dtype_,
        device,
        tp_rank_,
        tp_size_);
    INFINICORE_NN_MODULE_INIT(q_norm, q_lora_rank_, rms_norm_eps_, dtype_, device);
    INFINICORE_NN_MODULE_INIT(kv_norm, head_dim_, rms_norm_eps_, dtype_, device);

    wo_a_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "wo_a",
        num_attention_heads_ * head_dim_ / o_groups_,
        o_groups_ * o_lora_rank_,
        false,
        dtype_,
        device,
        tp_rank_,
        tp_size_);
    wo_b_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "wo_b",
        o_groups_ * o_lora_rank_,
        hidden_size_,
        quantization_method,
        false,
        dtype_,
        device,
        tp_rank_,
        tp_size_,
        rank_info.comm);
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

infinicore::Tensor DeepseekV4Attention::raw_cache_view_(const infinicore::Tensor &raw_cache) const {
    if (!raw_cache || raw_cache->ndim() != 2 || raw_cache->dtype() != infinicore::DataType::U8) {
        throw std::runtime_error("DeepseekV4Attention: sparse FlashMLA requires uint8 raw cache [blocks, page_bytes]");
    }
    const size_t page_size = infer_flashmla_raw_cache_page_size(raw_cache);
    auto *raw_ptr = const_cast<std::byte *>(raw_cache->data());
    return infinicore::Tensor::strided_from_blob(
        raw_ptr,
        {raw_cache->size(0), page_size, 1, kDsv4FlashMlaBytesPerToken},
        {static_cast<infinicore::Stride>(raw_cache->size(1)),
         static_cast<infinicore::Stride>(kDsv4FlashMlaBytesPerToken),
         static_cast<infinicore::Stride>(kDsv4FlashMlaBytesPerToken),
         1},
        infinicore::DataType::F8,
        raw_cache->device());
}

infinicore::Tensor DeepseekV4Attention::compute_sparse_attention(const infinicore::Tensor &q,
                                                                 const infinicore::Tensor &kv,
                                                                 size_t tokens) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    if (forward_context.kv_cache_vec.size() <= layer_idx_ || !forward_context.kv_cache_vec[layer_idx_]) {
        throw std::runtime_error("DeepseekV4Attention::compute_sparse_attention requires DeepSeek-V4 raw KV cache");
    }
    if (!forward_context.swa_attn_metadata.has_metadata()) {
        throw std::runtime_error("DeepseekV4Attention::compute_sparse_attention requires DeepSeek-V4 SWA metadata");
    }
    if (forward_context.swa_attn_metadata.swa_indices->ndim() != 2
        || forward_context.swa_attn_metadata.swa_indices->dtype() != infinicore::DataType::I32
        || forward_context.swa_attn_metadata.swa_indices->size(0) != tokens
        || forward_context.swa_attn_metadata.swa_topk_lengths->ndim() != 1
        || forward_context.swa_attn_metadata.swa_topk_lengths->dtype() != infinicore::DataType::I32
        || forward_context.swa_attn_metadata.swa_topk_lengths->size(0) != tokens
        || forward_context.swa_attn_metadata.raw_out_loc->ndim() != 1
        || forward_context.swa_attn_metadata.raw_out_loc->dtype() != infinicore::DataType::I32
        || forward_context.swa_attn_metadata.raw_out_loc->size(0) != tokens) {
        throw std::runtime_error("DeepseekV4Attention::forward_sparse_mla_ DeepSeek-V4 metadata shape mismatch");
    }

    auto &raw_cache = forward_context.kv_cache_vec[layer_idx_];
    const size_t page_size = infer_flashmla_raw_cache_page_size(raw_cache);
    infinicore::op::deepseek_v4::store_flashmla_raw_cache_(
        kv,
        raw_cache,
        forward_context.swa_attn_metadata.raw_out_loc,
        static_cast<int>(page_size));

    ASSERT(forward_context.sched_meta.sched_meta_vec.size() > 0);
    auto &sched_meta = forward_context.sched_meta.sched_meta_vec[0];
    auto cache4d = raw_cache_view_(raw_cache);
    auto indices3d = forward_context.swa_attn_metadata.swa_indices->view({tokens, 1, forward_context.swa_attn_metadata.swa_indices->size(forward_context.swa_attn_metadata.swa_indices->ndim() - 1)});

    auto local_attn_sink = attn_sink_;
    if (tp_size_ > 1) {
        local_attn_sink = attn_sink_->narrow({{0, tp_rank_ * num_local_attention_heads_, num_local_attention_heads_}});
    }
    auto q4d = q->view({tokens, 1, num_local_attention_heads_, head_dim_});
    auto [out4d, lse] = sparse_mla_attn_->forward(
        q4d,
        cache4d,
        indices3d,
        local_attn_sink,
        forward_context.swa_attn_metadata.swa_topk_lengths,
        sched_meta);
    (void)lse;
    return out4d->view({tokens, num_local_attention_heads_, head_dim_});
}

infinicore::Tensor DeepseekV4Attention::apply_output_projection_(const infinicore::Tensor &attn_out,
                                                                 size_t tokens) const {
    auto wo_a_in = attn_out->view({tokens, num_local_attention_heads_ * head_dim_ / num_local_groups_});
    auto wo_a_out = wo_a_->forward(wo_a_in);
    return wo_b_->forward(wo_a_out);
}

void DeepseekV4Attention::apply_rope_(const infinicore::Tensor &positions,
                                      infinicore::Tensor query,
                                      std::optional<infinicore::Tensor> key,
                                      bool inverse) const {
    if (!query || query->size(0) == 0) {
        return;
    }
    infinicore::op::deepseek_v4::fused_rope_(query, key, rope_freqs_cis_, positions, inverse);
}

infinicore::Tensor DeepseekV4Attention::forward(const infinicore::Tensor &positions,
                                                const infinicore::Tensor &hidden_states) const {
    if (hidden_states->ndim() != 2 || hidden_states->size(1) != hidden_size_) {
        throw std::runtime_error("DeepseekV4Attention::forward expects hidden_states [tokens, hidden]");
    }
    auto flat_positions = positions->view({positions->numel()});
    const size_t tokens = hidden_states->size(0);

    auto x = hidden_states;
    infinicore::Tensor q_lora, kv;
    {
        std::tie(q_lora, kv) = wqkv_a_->forward_split(x);
        q_lora = q_norm_->forward(q_lora);
        kv = kv_norm_->forward(kv);
    }

    infinicore::Tensor q, q_rope;
    {
        q = wq_b_->forward(q_lora)->view({tokens, num_local_attention_heads_, head_dim_});
        q = infinicore::op::deepseek_v4::rmsnorm_self(q, static_cast<float>(rms_norm_eps_));
        q_rope = q->narrow({{2, head_dim_ - qk_rope_head_dim_, qk_rope_head_dim_}});
    }

    auto kv_rope = kv->narrow({{1, head_dim_ - qk_rope_head_dim_, qk_rope_head_dim_}})->unsqueeze(1);

    apply_rope_(flat_positions, q_rope, kv_rope, false);

    auto attn_out = compute_sparse_attention(q, kv, tokens);

    auto out_rope = attn_out->narrow({{2, head_dim_ - qk_rope_head_dim_, qk_rope_head_dim_}});
    apply_rope_(flat_positions, out_rope, std::nullopt, true);

    return apply_output_projection_(attn_out, tokens);
}

void DeepseekV4Attention::process_weights_after_loading() {
    wqkv_a_->process_weights_after_loading();
    wq_b_->process_weights_after_loading();
    wo_a_->process_weights_after_loading();
    wo_b_->process_weights_after_loading();
}

void DeepseekV4Attention::reset_runtime_state() const {
    wqkv_a_->reset_runtime_state();
    wq_b_->reset_runtime_state();
    wo_a_->reset_runtime_state();
    wo_b_->reset_runtime_state();
}

} // namespace infinilm::models::deepseek_v4
