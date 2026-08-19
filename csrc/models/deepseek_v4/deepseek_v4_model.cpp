#include "deepseek_v4_model.hpp"

#include "infinicore/ops/deepseek_v4_embedding_and_hc_expand.hpp"
#include "infinicore/ops/deepseek_v4_hc_head.hpp"

#include <stdexcept>
#include <string>
#include <tuple>

namespace infinilm::models::deepseek_v4 {

DeepseekV4Model::DeepseekV4Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    const size_t hc_mult = model_config->get_or<size_t>("hc_mult", 4);
    const size_t hc_dim = hc_mult * hidden_size;
    hidden_size_ = hidden_size;
    hc_mult_ = hc_mult;
    rms_norm_eps_ = rms_norm_eps;
    hc_eps_ = model_config->get_or<double>("hc_eps", 1e-6);

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt, dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<DeepseekV4DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, rms_norm_eps, dtype, device);

    INFINICORE_NN_PARAMETER_INIT(hc_head_fn, ({hc_mult, hc_dim}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_head_base, ({hc_mult}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_head_scale, ({1}, infinicore::DataType::F32, device));
}

infinicore::Tensor expand_hc_stream(const infinicore::Tensor &hidden_states,
                                    size_t hc_mult) {
    const auto shape = hidden_states->shape();
    if (shape.size() != 2) {
        throw std::runtime_error("DeepseekV4MHC: expected hidden_states shape [tokens, hidden]");
    }
    const size_t ntoken = shape[0];
    const size_t hidden_size = shape[1];
    const auto strides = hidden_states->strides();
    if (strides.size() != 2) {
        throw std::runtime_error("DeepseekV4MHC: expected hidden_states strides [tokens, hidden]");
    }

    return hidden_states->as_strided(
                            {ntoken, hc_mult, hidden_size},
                            {strides[0], 0, strides[1]})
        ->contiguous();
}

infinicore::Tensor DeepseekV4Model::forward(const infinilm::InfinilmModel::Input &input) const {
    if (!input.input_ids.has_value() || !input.position_ids.has_value()) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4Model: input_ids and position_ids are required");
    }
    auto flat_input_ids = input.input_ids.value()->view({input.input_ids.value()->numel()});

    infinicore::Tensor hidden_states;
    bool use_embedding_and_expand_hc = true;
    if (use_embedding_and_expand_hc) {
        // 1000次 deepseek_v4_embedding+expand_hc_stream是 7 ms
        hidden_states = embedding_hc_expand_scratch_.get(
            {flat_input_ids->numel(), hc_mult_, hidden_size_},
            embed_tokens_->weight()->dtype(),
            embed_tokens_->weight()->device());
        infinicore::op::deepseek_v4_embedding_and_hc_expand_(
            hidden_states,
            flat_input_ids,
            embed_tokens_->weight(),
            static_cast<int64_t>(hc_mult_));
    } else {
        // 1000次 237 ms
        // hidden_states = embed_tokens_->forward(flat_input_ids);
        // 1000次 7 ms
        // 1000次 deepseek_v4_embedding+expand_hc_stream是 10 ms
        hidden_states = embed_tokens_->forward(flat_input_ids);
        if (hidden_states->ndim() != 2) {
            hidden_states = hidden_states->view({hidden_states->numel() / hidden_size_, hidden_size_});
        }
        hidden_states = expand_hc_stream(hidden_states, hc_mult_);
    }
    infinicore::Tensor prev_residual;
    infinicore::Tensor prev_post;
    infinicore::Tensor prev_comb;
    {
        // 1000 次 bs=1  fused=false  prefill =12481.360  decode 6040.812
        // 1000 次 bs=1  fused=true   prefill 8016.859  decode 5430.688

        // 1000 次 bs=32 fused=false  prefill =23688.123  decode 8762.537
        // 1000 次 bs=32 fused=true   prefill =16779.734  decode 7549.662

        for (const auto &layer : layers_) {
            auto [next_hidden_states, next_residual, next_post, next_comb] = layer->forward(input.position_ids.value(),
                                                                                            hidden_states,
                                                                                            flat_input_ids,
                                                                                            prev_residual,
                                                                                            prev_post,
                                                                                            prev_comb);
            hidden_states = next_hidden_states;
            prev_residual = next_residual;
            prev_post = next_post;
            prev_comb = next_comb;
        }
    }

    auto collapsed = hc_head_collapse_scratch_.get({hidden_states->size(0), hidden_size_},
                                                   hidden_states->dtype(),
                                                   hidden_states->device());
    infinicore::op::deepseek_v4_hc_head_(
        collapsed,
        hidden_states,
        hc_head_fn_,
        hc_head_scale_,
        hc_head_base_,
        rms_norm_eps_,
        hc_eps_);
    return norm_->forward(collapsed);
}

infinicore::Tensor DeepseekV4Model::embed_tokens(const infinicore::Tensor &input_ids) const {
    return embed_tokens_->forward(input_ids);
}

void DeepseekV4Model::process_weights_after_loading() {
    for (const auto &layer : layers_) {
        layer->process_weights_after_loading();
    }
}

void DeepseekV4Model::reset_runtime_state() const {
    for (const auto &layer : layers_) {
        layer->reset_runtime_state();
    }
}

} // namespace infinilm::models::deepseek_v4
