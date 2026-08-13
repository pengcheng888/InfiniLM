#include "deepseek_v4_decoder_layer.hpp"

#include "../../global_state/global_state.hpp"
#include "deepseek_v4_profile.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/ops/deepseek_v4_mhc_fused_post_pre.hpp"
#include "infinicore/ops/deepseek_v4_mhc_post.hpp"
#include "infinicore/ops/deepseek_v4_mhc_pre.hpp"

#include <stdexcept>
#include <string>
#include <tuple>

namespace infinilm::models::deepseek_v4 {

namespace {

void debug_dump_tensor(const infinicore::Tensor &tensor, size_t layer_idx, const std::string &name, bool enabled) {
    if (!enabled || !tensor) {
        return;
    }
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tensor->debug("/tmp/infinilm_dsv4_tp" + std::to_string(rank_info.tp_rank) + "_l" + std::to_string(layer_idx) + "_" + name + ".bin");
}

bool fused_mhc_post_pre_enabled() {
    const char *value = utils::env_value("INFINILM_DSV4_MHC_FUSED_POST_PRE");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    const std::string text(value);
    if (text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON") {
        return true;
    }
    if (text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "OFF" || text == "naive") {
        return false;
    }
    throw std::runtime_error("INFINILM_DSV4_MHC_FUSED_POST_PRE must be 0/1, true/false, or on/off");
}

} // namespace

thread_local DeepseekV4DecoderLayerSharedScratch DeepseekV4DecoderLayer::shared_scratch_;

DeepseekV4DecoderLayer::DeepseekV4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
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
    debug_dump_enabled_ = utils::env_flag_enabled("INFINILM_DSV4_DEBUG_DUMP");
    use_fused_mhc_post_pre_ = true; // fused_mhc_post_pre_enabled();

    INFINICORE_NN_MODULE_INIT(attn, model_config, layer_idx, device);
    compress_ratio_ = attn_->compress_ratio();
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

infinicore::Tensor DeepseekV4DecoderLayer::forward_naive(const infinicore::Tensor &positions,
                                                         infinicore::Tensor &hidden_states,
                                                         const infinicore::Tensor &input_ids) {
    (void)positions;
    if (hidden_states->ndim() != 3) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4DecoderLayer::forward_naive expects hidden_states [tokens, hc, hidden]");
    }
    const size_t token_count = hidden_states->size(0); // [tokens, hc, hidden]
    profile::ScopedLayerContext layer_context(profile::layer_type_from_compress_ratio(compress_ratio_));
    profile::ScopedTimer layer_timer(profile::Event::DecoderLayer, token_count);
    debug_dump_tensor(hidden_states, layer_idx_, "layer_input", debug_dump_enabled_);
    auto residual = hidden_states;
    auto attn_in = shared_scratch_.get_attn_in({token_count, hidden_size_},
                                               hidden_states->dtype(),
                                               hidden_states->device());
    auto attn_post = shared_scratch_.get_attn_post({token_count, hc_mult_},
                                                   infinicore::DataType::F32,
                                                   hidden_states->device());
    auto attn_comb = shared_scratch_.get_attn_comb({token_count, hc_mult_, hc_mult_},
                                                   infinicore::DataType::F32,
                                                   hidden_states->device());
    {
        profile::ScopedTimer timer(profile::Event::DecoderAttnHcPre, token_count);
        infinicore::op::deepseek_v4_mhc_pre_(
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
    }
    debug_dump_tensor(attn_in, layer_idx_, "attn_in", debug_dump_enabled_);
    debug_dump_tensor(attn_post, layer_idx_, "attn_post", debug_dump_enabled_);
    debug_dump_tensor(attn_comb, layer_idx_, "attn_comb", debug_dump_enabled_);
    {
        profile::ScopedTimer timer(profile::Event::DecoderAttnNorm, token_count);
        hidden_states = attn_norm_->forward(attn_in);
    }
    debug_dump_tensor(hidden_states, layer_idx_, "attn_normed", debug_dump_enabled_);

    hidden_states = attn_->forward(positions, hidden_states);

    debug_dump_tensor(hidden_states, layer_idx_, "attn_out", debug_dump_enabled_);
    auto attn_posted = infinicore::Tensor::empty(residual->shape(),
                                                 residual->dtype(),
                                                 residual->device());
    {
        profile::ScopedTimer timer(profile::Event::DecoderAttnHcPost, token_count);
        infinicore::op::deepseek_v4_mhc_post_(attn_posted, hidden_states, residual, attn_post, attn_comb);
    }
    debug_dump_tensor(attn_posted, layer_idx_, "attn_posted", debug_dump_enabled_);

    residual = attn_posted;
    auto ffn_in = shared_scratch_.get_ffn_in({token_count, hidden_size_},
                                             attn_posted->dtype(),
                                             attn_posted->device());
    auto ffn_post = shared_scratch_.get_ffn_post({token_count, hc_mult_},
                                                 infinicore::DataType::F32,
                                                 attn_posted->device());
    auto ffn_comb = shared_scratch_.get_ffn_comb({token_count, hc_mult_, hc_mult_},
                                                 infinicore::DataType::F32,
                                                 attn_posted->device());
    {
        profile::ScopedTimer timer(profile::Event::DecoderFfnHcPre, token_count);
        infinicore::op::deepseek_v4_mhc_pre_(
            ffn_in,
            ffn_post,
            ffn_comb,
            attn_posted,
            hc_ffn_fn_,
            hc_ffn_scale_,
            hc_ffn_base_,
            rms_norm_eps_,
            hc_eps_,
            hc_eps_,
            hc_sinkhorn_iters_);
    }
    debug_dump_tensor(ffn_in, layer_idx_, "ffn_in", debug_dump_enabled_);
    debug_dump_tensor(ffn_post, layer_idx_, "ffn_post", debug_dump_enabled_);
    debug_dump_tensor(ffn_comb, layer_idx_, "ffn_comb", debug_dump_enabled_);
    infinicore::Tensor ffn_normed;
    {
        profile::ScopedTimer timer(profile::Event::DecoderFfnNorm, token_count);
        ffn_normed = ffn_norm_->forward(ffn_in);
    }
    debug_dump_tensor(ffn_normed, layer_idx_, "ffn_normed", debug_dump_enabled_);

    infinicore::Tensor ffn_out;
    {
        profile::ScopedTimer timer(profile::Event::DecoderMoe, token_count);
        ffn_out = ffn_->forward(ffn_normed, input_ids);
    }

  // ffn_out = ffn_normed;

    hidden_states = infinicore::Tensor::empty(residual->shape(),
                                              residual->dtype(),
                                              residual->device());
    {
        profile::ScopedTimer timer(profile::Event::DecoderFfnHcPost, token_count);
        infinicore::op::deepseek_v4_mhc_post_(hidden_states, ffn_out, residual, ffn_post, ffn_comb);
    }
    debug_dump_tensor(hidden_states, layer_idx_, "layer_output", debug_dump_enabled_);
    return hidden_states;
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
DeepseekV4DecoderLayer::forward(const infinicore::Tensor &positions,
                                infinicore::Tensor &hidden_states,
                                const infinicore::Tensor &input_ids,
                                const infinicore::Tensor &prev_residual,
                                const infinicore::Tensor &prev_post,
                                const infinicore::Tensor &prev_comb) {
    if (!use_fused_mhc_post_pre_) {
        return {forward_naive(positions, hidden_states, input_ids), {}, {}, {}};
    }
    if (hidden_states->ndim() != 2 && hidden_states->ndim() != 3) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4DecoderLayer::forward expects hidden_states [tokens, hidden] or [tokens, hc, hidden]");
    }
    const size_t token_count = hidden_states->size(0);
    const auto dtype = hidden_states->dtype();
    const auto device = hidden_states->device();
    const infinicore::Shape hc_shape{token_count, hc_mult_, hidden_size_};

    profile::ScopedLayerContext layer_context(profile::layer_type_from_compress_ratio(compress_ratio_));
    profile::ScopedTimer layer_timer(profile::Event::DecoderLayer, token_count);
    debug_dump_tensor(hidden_states, layer_idx_, "layer_input", debug_dump_enabled_);

    auto attn_residual = shared_scratch_.get_attn_residual(hc_shape, dtype, device);
    auto attn_in = shared_scratch_.get_attn_in({token_count, hidden_size_}, dtype, device);
    auto attn_post = shared_scratch_.get_attn_post({token_count, hc_mult_}, infinicore::DataType::F32, device);
    auto attn_comb = shared_scratch_.get_attn_comb({token_count, hc_mult_, hc_mult_}, infinicore::DataType::F32, device);

    if (prev_residual) {
        if (hidden_states->ndim() != 2) {
            throw std::runtime_error("DeepseekV4DecoderLayer::forward expects 2D hidden_states when prev fused MHC state is present");
        }
        {
            profile::ScopedTimer timer(profile::Event::DecoderAttnHcPre, token_count);
            infinicore::op::deepseek_v4_mhc_fused_post_pre_(
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
                attn_norm_->weight(),
                attn_norm_->eps());
        }
    } else {
        if (hidden_states->ndim() != 3) {
            throw std::runtime_error("DeepseekV4DecoderLayer::forward expects first layer hidden_states [tokens, hc, hidden]");
        }
        attn_residual = hidden_states;
        {
            profile::ScopedTimer timer(profile::Event::DecoderAttnHcPre, token_count);
            infinicore::op::deepseek_v4_mhc_pre_(
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
        }
    }
    debug_dump_tensor(attn_in, layer_idx_, "attn_in", debug_dump_enabled_);
    debug_dump_tensor(attn_post, layer_idx_, "attn_post", debug_dump_enabled_);
    debug_dump_tensor(attn_comb, layer_idx_, "attn_comb", debug_dump_enabled_);

    {
        profile::ScopedTimer timer(profile::Event::DecoderAttnNorm, token_count);
        hidden_states = prev_residual ? attn_in : attn_norm_->forward(attn_in);
    }
    debug_dump_tensor(hidden_states, layer_idx_, "attn_normed", debug_dump_enabled_);

    hidden_states = attn_->forward(positions, hidden_states);
    debug_dump_tensor(hidden_states, layer_idx_, "attn_out", debug_dump_enabled_);

    auto ffn_residual = shared_scratch_.get_ffn_residual(hc_shape, dtype, device);
    auto ffn_in = shared_scratch_.get_ffn_in({token_count, hidden_size_}, dtype, device);
    auto ffn_post = shared_scratch_.get_ffn_post({token_count, hc_mult_}, infinicore::DataType::F32, device);
    auto ffn_comb = shared_scratch_.get_ffn_comb({token_count, hc_mult_, hc_mult_}, infinicore::DataType::F32, device);
    {
        profile::ScopedTimer timer(profile::Event::DecoderFfnHcPre, token_count);
        infinicore::op::deepseek_v4_mhc_fused_post_pre_(
            ffn_residual,
            ffn_post,
            ffn_comb,
            ffn_in,
            hidden_states,
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
            ffn_norm_->weight(),
            ffn_norm_->eps());
    }
    debug_dump_tensor(ffn_in, layer_idx_, "ffn_in", debug_dump_enabled_);
    debug_dump_tensor(ffn_post, layer_idx_, "ffn_post", debug_dump_enabled_);
    debug_dump_tensor(ffn_comb, layer_idx_, "ffn_comb", debug_dump_enabled_);

    infinicore::Tensor ffn_normed;
    {
        profile::ScopedTimer timer(profile::Event::DecoderFfnNorm, token_count);
        ffn_normed = ffn_in;
    }
    debug_dump_tensor(ffn_normed, layer_idx_, "ffn_normed", debug_dump_enabled_);

    infinicore::Tensor ffn_out;
    {
        profile::ScopedTimer timer(profile::Event::DecoderMoe, token_count);
        ffn_out = ffn_->forward(ffn_normed, input_ids);
    }
    // ffn_out = ffn_normed;

    debug_dump_tensor(ffn_out, layer_idx_, "layer_deferred_output", debug_dump_enabled_);
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
    const size_t token_count = hidden_states->size(0);
    auto output = shared_scratch_.get_attn_residual(residual->shape(), residual->dtype(), residual->device());
    {
        profile::ScopedTimer timer(profile::Event::DecoderFfnHcPost, token_count);
        infinicore::op::deepseek_v4_mhc_post_(output, hidden_states, residual, post, comb);
    }
    debug_dump_tensor(output, layer_idx_, "layer_output", debug_dump_enabled_);
    return output;
}

void DeepseekV4DecoderLayer::process_weights_after_loading() {
    attn_->process_weights_after_loading();
    ffn_->process_weights_after_loading();
    shared_scratch_.preallocate_scratch(hidden_size_, hc_mult_, dtype_, device_);
}

void DeepseekV4DecoderLayer::reset_runtime_state() const {
    attn_->reset_runtime_state();
    ffn_->reset_runtime_state();
}

} // namespace infinilm::models::deepseek_v4
