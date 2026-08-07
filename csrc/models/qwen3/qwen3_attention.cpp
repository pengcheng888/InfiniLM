#include "qwen3_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "infinicore/ops/qwen3_fused_qk_norm_rope.hpp"
#include "infinicore/ops/qwen3_mha_kvcache.hpp"
#include "infinicore/ops/qwen3_mha_varlen.hpp"
#include "infinicore/ops/qwen3_store_kvcache.hpp"
#include <cmath>
#include <optional>
#include <stdexcept>

namespace infinilm::models::qwen3 {

Qwen3Attention::Qwen3Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               size_t layer_idx,
                               const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    max_position_embeddings_ = model_config->get<size_t>("max_position_embeddings");
    rotary_dim_ = model_config->get_rotary_dim();
    rms_norm_eps_ = static_cast<float>(model_config->get<double>("rms_norm_eps"));
    rope_theta_ = static_cast<float>(model_config->get_or<double>("rope_theta", 1000000.0));
    rope_factor_ = 1.0f;
    rope_low_ = 0.0f;
    rope_high_ = 0.0f;
    rope_attention_factor_ = 1.0f;
    is_neox_ = true;

    const auto &dtype = model_config->get_dtype();
    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    bool use_bias = model_config->get_or<bool>("attention_bias", false);
    bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        throw std::runtime_error("infinilm::models::qwen3::Qwen3Attention: static attention is not supported for qwen3");
    }

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();
    if ((total_num_heads % tp_size) != 0) {
        throw std::runtime_error("infinilm::models::qwen3::Qwen3Attention: num_attention_heads must be divisible by tp_size");
    }
    if ((total_num_kv_heads < static_cast<size_t>(tp_size)) || ((total_num_kv_heads % tp_size) != 0)) {
        throw std::runtime_error("infinilm::models::qwen3::Qwen3Attention: num_key_value_heads must be divisible by tp_size for dense qwen3");
    }

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads / tp_size;
    q_size_ = num_attention_heads_ * head_dim_;
    k_size_ = num_key_value_heads_ * head_dim_;
    v_size_ = num_key_value_heads_ * head_dim_;
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) {
        this->register_parameter(n, std::move(p));
    };
    qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype, device, rank_info);
    o_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, rms_norm_eps_, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, rms_norm_eps_, dtype, device);

    infinilm::layers::attention::init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor Qwen3Attention::forward(const infinicore::Tensor &positions,
                                           const infinicore::Tensor &hidden_states) const {
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Qwen3Attention::prepare_position_ids_(const infinicore::Tensor &position_ids,
                                                         size_t seq_len) const {
    auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        return pos_narrowed->view({seq_len});
    }
    if (pos_shape.size() == 1) {
        return position_ids;
    }
    throw std::runtime_error("infinilm::models::qwen3::Qwen3Attention: unexpected position_ids shape");
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Qwen3Attention::do_kv_cache_update(
    const infinicore::Tensor key,
    const infinicore::Tensor value,
    infinicore::Tensor &kv_cache,
    const infinicore::Tensor slot_mapping) const {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    auto k_cache_flat = k_cache_layer->view({k_cache_layer->shape()[0] * k_cache_layer->shape()[1], num_key_value_heads_, head_dim_});
    auto v_cache_flat = v_cache_layer->view({v_cache_layer->shape()[0] * v_cache_layer->shape()[1], num_key_value_heads_, head_dim_});
    infinicore::op::qwen3_store_kvcache_(key, value, k_cache_flat, v_cache_flat, slot_mapping);

    return {k_cache_layer, v_cache_layer};
}

infinicore::Tensor Qwen3Attention::caculate_attention(
    const infinicore::Tensor &query,
    const infinicore::Tensor &key_cache,
    const infinicore::Tensor &value_cache,
    const infinilm::global_state::AttentionMetadata &attn_metadata) const {
    ASSERT(attn_metadata.total_sequence_lengths.has_value());
    ASSERT(attn_metadata.input_offsets.has_value());
    ASSERT(attn_metadata.cu_seqlens.has_value());
    ASSERT(attn_metadata.block_tables.has_value());

    size_t seq_len = query->shape()[0];
    bool is_prefill = (seq_len != attn_metadata.total_sequence_lengths.value()->shape()[0]);
    infinicore::Tensor attn_output;
    if (is_prefill) {
        attn_output = infinicore::op::qwen3_mha_varlen(
            query,
            key_cache,
            value_cache,
            attn_metadata.input_offsets.value(),
            attn_metadata.cu_seqlens.value(),
            attn_metadata.block_tables.value(),
            static_cast<int>(max_position_embeddings_),
            static_cast<int>(max_position_embeddings_),
            std::nullopt,
            scale_);
    } else {
        auto q_for_fa = query->view({seq_len, 1, num_attention_heads_, head_dim_});
        auto attn_out_4d = infinicore::op::qwen3_mha_kvcache(
            q_for_fa,
            key_cache,
            value_cache,
            attn_metadata.total_sequence_lengths.value(),
            attn_metadata.block_tables.value(),
            std::nullopt,
            scale_);
        attn_output = attn_out_4d->view({seq_len, num_attention_heads_, head_dim_});
    }

    return attn_output->view({1, seq_len, num_attention_heads_ * head_dim_});
}

infinicore::Tensor Qwen3Attention::forward_paged_(const infinicore::Tensor &position_ids,
                                                  const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);

    auto hidden_states_mutable = hidden_states;
    auto qkv = qkv_proj_->forward(hidden_states_mutable)->view({seq_len, q_size_ + k_size_ + v_size_});
    auto pos_ids_for_rope = prepare_position_ids_(position_ids, seq_len);

    infinicore::op::qwen3_fused_qk_norm_rope_(
        qkv,
        static_cast<int>(num_attention_heads_),
        static_cast<int>(num_key_value_heads_),
        static_cast<int>(num_key_value_heads_),
        static_cast<int>(head_dim_),
        rms_norm_eps_,
        q_norm_->weight(),
        k_norm_->weight(),
        rope_theta_,
        is_neox_,
        pos_ids_for_rope,
        rope_factor_,
        rope_low_,
        rope_high_,
        rope_attention_factor_,
        static_cast<int>(rotary_dim_));

    auto q = qkv->narrow({{1, 0, q_size_}})->contiguous()->view({seq_len, num_attention_heads_, head_dim_});
    auto k = qkv->narrow({{1, q_size_, k_size_}})->contiguous()->view({seq_len, num_key_value_heads_, head_dim_});
    auto v = qkv->narrow({{1, q_size_ + k_size_, v_size_}})->contiguous()->view({seq_len, num_key_value_heads_, head_dim_});

    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.attn_metadata;
    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
    ASSERT(attn_metadata.total_sequence_lengths.has_value());
    ASSERT(attn_metadata.input_offsets.has_value());
    ASSERT(attn_metadata.cu_seqlens.has_value());
    ASSERT(attn_metadata.block_tables.has_value());
    ASSERT(attn_metadata.slot_mapping.has_value());

    auto [k_cache_layer, v_cache_layer] = do_kv_cache_update(k, v, kv_cache, attn_metadata.slot_mapping.value());

    auto attn_output = caculate_attention(q, k_cache_layer, v_cache_layer, attn_metadata);
    
    return o_proj_->forward(attn_output);
}

} // namespace infinilm::models::qwen3
