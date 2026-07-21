#include "deepseek_v4_decoder_layer.hpp"

#include "../../global_state/global_state.hpp"
#include "deepseek_v4_profile.hpp"
#include "infinicore/ops/deepseek_v4_mhc.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {

namespace {

bool debug_dump_enabled() {
    const char *value = std::getenv("INFINILM_DSV4_DEBUG_DUMP");
    if (value == nullptr) {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

void debug_dump_tensor(const infinicore::Tensor &tensor, size_t layer_idx, const std::string &name) {
    if (!debug_dump_enabled() || !tensor) {
        return;
    }
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tensor->debug("/tmp/infinilm_dsv4_tp" + std::to_string(rank_info.tp_rank) + "_l" + std::to_string(layer_idx) + "_" + name + ".bin");
}

} // namespace

DeepseekV4DecoderLayer::DeepseekV4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               size_t layer_idx,
                                               const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    const size_t hc_mult = model_config->get_or<size_t>("hc_mult", 4);
    const size_t mix_hc = (2 + hc_mult) * hc_mult;
    const size_t hc_dim = hc_mult * hidden_size;
    hidden_size_ = hidden_size;
    hc_mult_ = hc_mult;
    rms_norm_eps_ = rms_norm_eps;
    hc_eps_ = model_config->get_or<double>("hc_eps", 1e-6);
    hc_sinkhorn_iters_ = static_cast<int>(model_config->get_or<size_t>("hc_sinkhorn_iters", 20));

    INFINICORE_NN_MODULE_INIT(attn, model_config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(ffn, model_config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(attn_norm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(ffn_norm, hidden_size, rms_norm_eps, dtype, device);

    INFINICORE_NN_PARAMETER_INIT(hc_attn_fn, ({mix_hc, hc_dim}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_fn, ({mix_hc, hc_dim}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_attn_base, ({mix_hc}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_base, ({mix_hc}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_attn_scale, ({3}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_scale, ({3}, infinicore::DataType::F32, device));
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
DeepseekV4DecoderLayer::forward(const infinicore::Tensor &positions,
                                infinicore::Tensor &hidden_states,
                                infinicore::Tensor &residual,
                                const infinicore::Tensor &input_ids) {
    (void)positions;
    if (hidden_states->ndim() != 3) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4DecoderLayer::forward expects hidden_states [tokens, hc, hidden]");
    }
    const size_t token_count = hidden_states->size(0);
    profile::ScopedTimer layer_timer(profile::Event::DecoderLayer, token_count);
    debug_dump_tensor(hidden_states, layer_idx_, "layer_input");
    residual = hidden_states;
    auto attn_in = infinicore::Tensor::empty({hidden_states->size(0), hidden_size_}, hidden_states->dtype(), hidden_states->device());
    auto attn_post = infinicore::Tensor::empty({hidden_states->size(0), hc_mult_}, infinicore::DataType::F32, hidden_states->device());
    auto attn_comb = infinicore::Tensor::empty({hidden_states->size(0), hc_mult_, hc_mult_}, infinicore::DataType::F32, hidden_states->device());
    {
        profile::ScopedTimer timer(profile::Event::DecoderAttnHcPre, token_count);
        infinicore::op::deepseek_v4_mhc_pre_naive_(
            attn_in,
            attn_post,
            attn_comb,
            hidden_states,
            hc_attn_fn_,
            hc_attn_scale_,
            hc_attn_base_,
            rms_norm_eps_,
            hc_eps_,
            hc_sinkhorn_iters_);
    }
    debug_dump_tensor(attn_in, layer_idx_, "attn_in");
    debug_dump_tensor(attn_post, layer_idx_, "attn_post");
    debug_dump_tensor(attn_comb, layer_idx_, "attn_comb");
    {
        profile::ScopedTimer timer(profile::Event::DecoderAttnNorm, token_count);
        hidden_states = attn_norm_->forward(attn_in);
    }
    debug_dump_tensor(hidden_states, layer_idx_, "attn_normed");

    // Attention is intentionally skipped until DeepseekV4Attention::forward is implemented.
    auto attn_posted = infinicore::Tensor::empty(residual->shape(), residual->dtype(), residual->device());
    {
        profile::ScopedTimer timer(profile::Event::DecoderAttnHcPost, token_count);
        infinicore::op::deepseek_v4_mhc_post_naive_(attn_posted, hidden_states, residual, attn_post, attn_comb);
    }
    debug_dump_tensor(attn_posted, layer_idx_, "attn_posted");

    residual = attn_posted;
    auto ffn_in = infinicore::Tensor::empty({attn_posted->size(0), hidden_size_}, attn_posted->dtype(), attn_posted->device());
    auto ffn_post = infinicore::Tensor::empty({attn_posted->size(0), hc_mult_}, infinicore::DataType::F32, attn_posted->device());
    auto ffn_comb = infinicore::Tensor::empty({attn_posted->size(0), hc_mult_, hc_mult_}, infinicore::DataType::F32, attn_posted->device());
    {
        profile::ScopedTimer timer(profile::Event::DecoderFfnHcPre, token_count);
        infinicore::op::deepseek_v4_mhc_pre_naive_(
            ffn_in,
            ffn_post,
            ffn_comb,
            attn_posted,
            hc_ffn_fn_,
            hc_ffn_scale_,
            hc_ffn_base_,
            rms_norm_eps_,
            hc_eps_,
            hc_sinkhorn_iters_);
    }
    debug_dump_tensor(ffn_in, layer_idx_, "ffn_in");
    debug_dump_tensor(ffn_post, layer_idx_, "ffn_post");
    debug_dump_tensor(ffn_comb, layer_idx_, "ffn_comb");
    infinicore::Tensor ffn_normed;
    {
        profile::ScopedTimer timer(profile::Event::DecoderFfnNorm, token_count);
        ffn_normed = ffn_norm_->forward(ffn_in);
    }
    debug_dump_tensor(ffn_normed, layer_idx_, "ffn_normed");

    infinicore::Tensor ffn_out;
    {
        profile::ScopedTimer timer(profile::Event::DecoderMoe, token_count);
        ffn_out = ffn_->forward(ffn_normed, input_ids);
    }
    // ffn_out = ffn_normed; // skip ffn_ for test.

    hidden_states = infinicore::Tensor::empty(residual->shape(), residual->dtype(), residual->device());
    {
        profile::ScopedTimer timer(profile::Event::DecoderFfnHcPost, token_count);
        infinicore::op::deepseek_v4_mhc_post_naive_(hidden_states, ffn_out, residual, ffn_post, ffn_comb);
    }
    debug_dump_tensor(hidden_states, layer_idx_, "layer_output");
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor DeepseekV4DecoderLayer::forward(const infinicore::Tensor &positions,
                                                   infinicore::Tensor &hidden_states,
                                                   const infinicore::Tensor &input_ids) {
    infinicore::Tensor residual;
    std::tie(hidden_states, residual) = forward(positions, hidden_states, residual, input_ids);
    return hidden_states;
}

void DeepseekV4DecoderLayer::process_weights_after_loading() {
    attn_->process_weights_after_loading();
    ffn_->process_weights_after_loading();
}

void DeepseekV4DecoderLayer::reset_runtime_state() const {
    attn_->reset_runtime_state();
    ffn_->reset_runtime_state();
}

} // namespace infinilm::models::deepseek_v4
