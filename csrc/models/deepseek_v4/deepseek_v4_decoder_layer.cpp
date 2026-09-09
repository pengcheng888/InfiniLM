#include "deepseek_v4_decoder_layer.hpp"

#include "infinicore/ops/deepseek_v4/mhc_fused_post_pre.hpp"
#include "infinicore/ops/deepseek_v4/mhc_post.hpp"
#include "infinicore/ops/deepseek_v4/mhc_pre.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v4 {

thread_local DeepseekV4DecoderLayerSharedScratch DeepseekV4DecoderLayer::shared_scratch_;

DeepseekV4DecoderLayer::DeepseekV4DecoderLayer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    const size_t hc_mult = model_config->get_or<size_t>("hc_mult", 4);
    const size_t mix_hc = (2 + hc_mult) * hc_mult;
    const size_t hc_dim = hc_mult * hidden_size;

    dtype_ = dtype;
    device_ = device;
    hidden_size_ = hidden_size;
    hc_mult_ = hc_mult;
    rms_norm_eps_ = rms_norm_eps;
    hc_eps_ = model_config->get_or<double>("hc_eps", 1e-6);
    hc_sinkhorn_iters_ = static_cast<int>(model_config->get_or<size_t>("hc_sinkhorn_iters", 20));
    is_last_layer_ = (layer_idx + 1 == num_hidden_layers);

    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    moe_ = this->register_module<DeepseekV4MoE>("mlp", model_config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);

    INFINICORE_NN_PARAMETER_INIT(hc_attn_fn, ({mix_hc, hc_dim}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_fn, ({mix_hc, hc_dim}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_attn_base, ({mix_hc}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_base, ({mix_hc}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_attn_scale, ({3}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_scale, ({3}, infinicore::DataType::F32, device));
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, infinicore::Tensor> DeepseekV4DecoderLayer::forward(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    const infinicore::Tensor &input_ids,
    const infinicore::Tensor &prev_residual,
    const infinicore::Tensor &prev_post,
    const infinicore::Tensor &prev_comb) const {
    if (hidden_states->ndim() != 2 && hidden_states->ndim() != 3) {
        throw std::runtime_error("DeepseekV4DecoderLayer::forward expects hidden_states [tokens, hidden] or [tokens, hc, hidden]");
    }
    const size_t token_count = hidden_states->size(0);
    const auto dtype = hidden_states->dtype();
    const auto device = hidden_states->device();
    const infinicore::Shape hc_shape{token_count, hc_mult_, hidden_size_};

    auto attn_residual = shared_scratch_.get_attn_residual(hc_shape, dtype, device);
    auto attn_in = shared_scratch_.get_attn_in({token_count, hidden_size_}, dtype, device);
    auto attn_post = shared_scratch_.get_attn_post({token_count, hc_mult_}, infinicore::DataType::F32, device);
    auto attn_comb = shared_scratch_.get_attn_comb({token_count, hc_mult_, hc_mult_}, infinicore::DataType::F32, device);

    if (prev_residual) {
        if (hidden_states->ndim() != 2) {
            throw std::runtime_error("DeepseekV4DecoderLayer::forward expects 2D hidden_states when deferred MHC state is present");
        }
        infinicore::op::deepseek_v4::mhc_fused_post_pre_(
            attn_residual,
            attn_post,
            attn_comb,
            attn_in,
            hidden_states,
            prev_residual,
            prev_post,
            prev_comb,
            hc_attn_fn_,
            hc_attn_scale_,
            hc_attn_base_,
            rms_norm_eps_,
            hc_eps_,
            hc_eps_,
            2.0,
            hc_sinkhorn_iters_,
            input_layernorm_->weight(),
            input_layernorm_->eps());
    } else {
        if (hidden_states->ndim() != 3) {
            throw std::runtime_error("DeepseekV4DecoderLayer::forward expects first layer hidden_states [tokens, hc, hidden]");
        }
        attn_residual = hidden_states;
        infinicore::op::deepseek_v4::mhc_pre_(
            attn_in,
            attn_post,
            attn_comb,
            hidden_states,
            hc_attn_fn_,
            hc_attn_scale_,
            hc_attn_base_,
            rms_norm_eps_,
            hc_eps_,
            hc_eps_,
            hc_sinkhorn_iters_);
        attn_in = input_layernorm_->forward(attn_in);
    }

    auto attn_out = self_attn_->forward(positions, attn_in);

    auto ffn_residual = shared_scratch_.get_ffn_residual(hc_shape, dtype, device);
    auto ffn_in = shared_scratch_.get_ffn_in({token_count, hidden_size_}, dtype, device);
    auto ffn_post = shared_scratch_.get_ffn_post({token_count, hc_mult_}, infinicore::DataType::F32, device);
    auto ffn_comb = shared_scratch_.get_ffn_comb({token_count, hc_mult_, hc_mult_}, infinicore::DataType::F32, device);
    infinicore::op::deepseek_v4::mhc_fused_post_pre_(
        ffn_residual,
        ffn_post,
        ffn_comb,
        ffn_in,
        attn_out,
        attn_residual,
        attn_post,
        attn_comb,
        hc_ffn_fn_,
        hc_ffn_scale_,
        hc_ffn_base_,
        rms_norm_eps_,
        hc_eps_,
        hc_eps_,
        2.0,
        hc_sinkhorn_iters_,
        post_attention_layernorm_->weight(),
        post_attention_layernorm_->eps());

    auto ffn_out = moe_->forward(ffn_in, input_ids);
    if (is_last_layer_) {
        return {complete_deferred_hc_post(ffn_out, ffn_residual, ffn_post, ffn_comb), {}, {}, {}};
    }
    return {ffn_out, ffn_residual, ffn_post, ffn_comb};
}

infinicore::Tensor DeepseekV4DecoderLayer::complete_deferred_hc_post(const infinicore::Tensor &hidden_states,
                                                                     const infinicore::Tensor &residual,
                                                                     const infinicore::Tensor &post,
                                                                     const infinicore::Tensor &comb) const {
    if (!residual) {
        return hidden_states;
    }
    auto output = shared_scratch_.get_attn_residual(residual->shape(), residual->dtype(), residual->device());
    infinicore::op::deepseek_v4::mhc_post_(output, hidden_states, residual, post, comb);
    return output;
}

void DeepseekV4DecoderLayer::process_weights_after_loading() {
    self_attn_->process_weights_after_loading();
    if (moe_) {
        moe_->process_weights_after_loading();
    }
    shared_scratch_.preallocate_scratch(hidden_size_, hc_mult_, dtype_, device_);
}

void DeepseekV4DecoderLayer::reset_runtime_state() const {
    self_attn_->reset_runtime_state();
    if (moe_) {
        moe_->reset_runtime_state();
    }
}

} // namespace infinilm::models::deepseek_v4
