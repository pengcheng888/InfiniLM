#include "glm4_moe_lite_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../deepseek_v4/deepseek_v4_rope.hpp"

#include "infinicore/ops.hpp"
#include "infinicore/ops/baddbmm.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/deepseek_v4_fused_rope.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::glm4_moe_lite {

Glm4MoeLiteAttention::Glm4MoeLiteAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t layer_idx,
                                           const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto dtype = model_config->get_dtype();
    const auto quantization_method = model_config->get_quantization_method();
    hidden_size_ = model_config->get<size_t>("hidden_size");
    num_attention_heads_ = model_config->get<size_t>("num_attention_heads");
    q_lora_rank_ = model_config->get<size_t>("q_lora_rank");
    kv_lora_rank_ = model_config->get<size_t>("kv_lora_rank");
    qk_nope_head_dim_ = model_config->get<size_t>("qk_nope_head_dim");
    qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    v_head_dim_ = model_config->get<size_t>("v_head_dim");
    qk_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
    rms_norm_eps_ = model_config->get<double>("rms_norm_eps");

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    if (tp_size_ == 0 || num_attention_heads_ % tp_size_ != 0) {
        throw std::runtime_error("Glm4MoeLiteAttention: num_attention_heads must be divisible by tp_size");
    }
    num_local_attention_heads_ = num_attention_heads_ / tp_size_;
    softmax_scale_ = 1.0 / std::sqrt(static_cast<double>(qk_head_dim_));
    const size_t latent_dim = kv_lora_rank_ + qk_rope_head_dim_;

    auto register_qkv_a_param = [this](const std::string &name, infinicore::nn::Parameter param) {
        this->register_parameter(name, std::move(param));
    };
    qkv_a_proj_ = std::make_shared<infinilm::layers::linear::FusedReplicatedLinear>(
        hidden_size_,
        q_lora_rank_,
        kv_lora_rank_ + qk_rope_head_dim_,
        "q_a_proj",
        "kv_a_proj_with_mqa",
        register_qkv_a_param,
        quantization_method,
        false,
        dtype,
        device);

    q_b_proj_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "q_b_proj",
        q_lora_rank_,
        num_attention_heads_ * qk_head_dim_,
        quantization_method,
        false,
        dtype,
        device,
        tp_rank_,
        tp_size_);

    kv_b_proj_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "kv_b_proj",
        kv_lora_rank_,
        num_attention_heads_ * (qk_nope_head_dim_ + v_head_dim_),
        quantization_method,
        false,
        dtype,
        device,
        tp_rank_,
        tp_size_,
        static_cast<int>(num_attention_heads_));

    o_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "o_proj",
        num_attention_heads_ * v_head_dim_,
        hidden_size_,
        quantization_method,
        false,
        dtype,
        device,
        tp_rank_,
        tp_size_,
        rank_info.comm);

    mla_attn_ = std::make_shared<infinilm::layers::mla_attention::MLAAttentionLayer>(
        num_local_attention_heads_,
        latent_dim,
        static_cast<float>(softmax_scale_),
        1,
        layer_idx_,
        kv_lora_rank_,
        infinicore::Tensor(),
        infinicore::Tensor(),
        infinilm::backends::AttentionBackend::FLASHMLA);

    INFINICORE_NN_MODULE_INIT(q_a_layernorm, q_lora_rank_, rms_norm_eps_, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_a_layernorm, kv_lora_rank_, rms_norm_eps_, dtype, device);

    const size_t max_position_embeddings = model_config->get_or<size_t>("max_position_embeddings", 131072);
    const double rope_theta = model_config->get_or<double>("rope_theta", 1000000.0);
    rope_freqs_cis_ = infinilm::models::deepseek_v4::build_deepseek_v4_rope_freqs_cis(
        qk_rope_head_dim_,
        max_position_embeddings,
        false,
        rope_theta,
        rope_theta,
        1.0,
        32.0,
        1.0,
        0,
        device);
}

void Glm4MoeLiteAttention::apply_rope_(const infinicore::Tensor &positions,
                                       infinicore::Tensor query,
                                       infinicore::Tensor key) const {
    if (query->numel() == 0) {
        return;
    }
    infinicore::op::deepseek_v4_fused_rope_(query, key, rope_freqs_cis_, positions, false);
}

infinicore::Tensor Glm4MoeLiteAttention::forward(const infinicore::Tensor &positions,
                                                 const infinicore::Tensor &hidden_states) const {
    const auto hidden_shape = hidden_states->shape();
    if (hidden_shape.empty() || hidden_shape.back() != hidden_size_) {
        throw std::runtime_error("Glm4MoeLiteAttention::forward expects hidden size in the last dimension");
    }
    const bool restore_3d_shape = hidden_shape.size() == 3;
    const size_t tokens = hidden_states->numel() / hidden_size_;
    const size_t latent_dim = kv_lora_rank_ + qk_rope_head_dim_;
    auto x = hidden_states->view({tokens, hidden_size_});

    infinicore::Tensor q_lora;
    infinicore::Tensor kv_latent;
    std::tie(q_lora, kv_latent) = qkv_a_proj_->forward_split(x);
    q_lora = q_a_layernorm_->forward(q_lora);

    auto kv_c = kv_latent->narrow({{1, 0, kv_lora_rank_}});
    auto k_pe = kv_latent->narrow({{1, kv_lora_rank_, qk_rope_head_dim_}});
    kv_c = kv_a_layernorm_->forward(kv_c);

    auto q = q_b_proj_->forward(q_lora)->view({tokens, num_local_attention_heads_, qk_head_dim_});
    auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}});
    auto q_pe = q->narrow({{2, qk_nope_head_dim_, qk_rope_head_dim_}});
    auto k_pe_for_rope = k_pe->unsqueeze(1);
    apply_rope_(positions->view({positions->numel()}), q_pe, k_pe_for_rope);

    auto q_nope_by_head = q_nope->permute({1, 0, 2})->contiguous();
    auto q_nope_out_input = infinicore::Tensor::empty(
        {num_local_attention_heads_, tokens, kv_lora_rank_},
        q_nope->dtype(),
        q_nope->device());
    auto q_nope_out = infinicore::op::baddbmm(q_nope_out_input, q_nope_by_head, w_kc_, 0.0f, 1.0f)
                          ->permute({1, 0, 2})
                          ->contiguous();
    auto q_flash = infinicore::op::cat({q_nope_out, q_pe}, 2)->view({1, tokens, num_local_attention_heads_, latent_dim});

    auto [attn_latent_4d, lse] = mla_attn_->forward_mqa(q_flash, kv_c, k_pe);
    (void)lse;

    auto attn_by_head = attn_latent_4d->view({tokens, num_local_attention_heads_, kv_lora_rank_})
                            ->permute({1, 0, 2})
                            ->contiguous();
    auto out_input = infinicore::Tensor::empty(
        {num_local_attention_heads_, tokens, v_head_dim_},
        attn_by_head->dtype(),
        attn_by_head->device());
    auto out_by_head = infinicore::op::baddbmm(out_input, attn_by_head, w_vc_, 0.0f, 1.0f);
    auto out_flat = out_by_head->permute({1, 0, 2})
                        ->contiguous()
                        ->view({tokens, num_local_attention_heads_ * v_head_dim_});
    auto output = o_proj_->forward(out_flat);
    return restore_3d_shape ? output->view(hidden_shape) : output;
}

void Glm4MoeLiteAttention::process_weights_after_loading() {
    qkv_a_proj_->process_weights_after_loading();
    q_b_proj_->process_weights_after_loading();
    kv_b_proj_->process_weights_after_loading();
    o_proj_->process_weights_after_loading();

    auto kv_b_weight = kv_b_proj_->weight();
    if (!kv_b_weight || kv_b_weight->ndim() != 2) {
        throw std::runtime_error("Glm4MoeLiteAttention: kv_b_proj.weight must be a 2D tensor after loading");
    }
    if (kv_b_weight->size(0) != num_local_attention_heads_ * (qk_nope_head_dim_ + v_head_dim_)
        || kv_b_weight->size(1) != kv_lora_rank_) {
        throw std::runtime_error("Glm4MoeLiteAttention: unexpected kv_b_proj.weight shape");
    }
    auto kv_b_view = kv_b_weight->view({num_local_attention_heads_, qk_nope_head_dim_ + v_head_dim_, kv_lora_rank_});
    w_kc_ = kv_b_view->narrow({{1, 0, qk_nope_head_dim_}})->contiguous();
    w_vc_ = kv_b_view->narrow({{1, qk_nope_head_dim_, v_head_dim_}})->permute({0, 2, 1})->contiguous();
}

void Glm4MoeLiteAttention::reset_runtime_state() const {
    qkv_a_proj_->reset_runtime_state();
    q_b_proj_->reset_runtime_state();
    kv_b_proj_->reset_runtime_state();
    o_proj_->reset_runtime_state();
}

} // namespace infinilm::models::glm4_moe_lite
