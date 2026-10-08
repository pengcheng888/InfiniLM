#include "deepseek_v4_model.hpp"

#include "../../global_state/global_state.hpp"

#include "infinicore/ops/deepseek_v4/embedding_and_hc_expand.hpp"
#include "infinicore/ops/deepseek_v4/hc_head.hpp"
#include "infinicore/ops/distributed/allgather.hpp"
#include "infinicore/ops/distributed/send_recv.hpp"

#include <stdexcept>
#include <string>

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
    dtype_ = dtype;
    rms_norm_eps_ = rms_norm_eps;
    hc_eps_ = model_config->get_or<double>("hc_eps", 1e-6);

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    pp_size_ = static_cast<size_t>(rank_info.pp_size);
    pp_stage_ = static_cast<size_t>(rank_info.pp_stage);
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
    device_ = device;
    const size_t local_layer_begin = num_hidden_layers * pp_stage_ / pp_size_;
    const size_t local_layer_end = num_hidden_layers * (pp_stage_ + 1) / pp_size_;
    if (pp_size_ == 0 || pp_stage_ >= pp_size_ || hidden_size_ % tp_size_ != 0) {
        throw std::runtime_error("DeepseekV4Model: invalid PP/TP configuration");
    }

    if (is_first_pp_stage()) {
        INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt, dtype, device);
    }
    layers_.reserve(local_layer_end - local_layer_begin);
    for (size_t i = local_layer_begin; i < local_layer_end; ++i) {
        layers_.push_back(this->register_module<DeepseekV4DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }
    if (is_last_pp_stage()) {
        INFINICORE_NN_MODULE_INIT(norm, hidden_size, rms_norm_eps, dtype, device);
        INFINICORE_NN_PARAMETER_INIT(hc_head_fn, ({hc_mult, hc_dim}, infinicore::DataType::F32, device));
        INFINICORE_NN_PARAMETER_INIT(hc_head_base, ({hc_mult}, infinicore::DataType::F32, device));
        INFINICORE_NN_PARAMETER_INIT(hc_head_scale, ({1}, infinicore::DataType::F32, device));
    }
}

infinicore::Tensor DeepseekV4Model::forward(const infinilm::InfinilmModel::Input &input) const {
    if (!input.input_ids.has_value() || !input.position_ids.has_value()) {
        throw std::runtime_error("DeepseekV4Model: input_ids and position_ids are required");
    }
    auto flat_input_ids = input.input_ids.value()->view({input.input_ids.value()->numel()});

    infinicore::Tensor hidden_states;
    if (is_first_pp_stage()) {
        hidden_states = embedding_hc_expand_scratch_.get(
            {flat_input_ids->numel(), hc_mult_, hidden_size_},
            embed_tokens_->weight()->dtype(),
            embed_tokens_->weight()->device());
        infinicore::op::deepseek_v4::embedding_and_hc_expand_(
            hidden_states,
            flat_input_ids,
            embed_tokens_->weight(),
            static_cast<int64_t>(hc_mult_));
    } else {
        hidden_states = recv_hidden_states(flat_input_ids->numel());
    }

    for (const auto &layer : layers_) {
        hidden_states = layer->forward(
            input.position_ids.value(),
            hidden_states,
            flat_input_ids);
    }

    if (!is_last_pp_stage()) {
        send_hidden_states(hidden_states);
        return hidden_states;
    }

    auto collapsed = hc_head_collapse_scratch_.get(
        {hidden_states->size(0), hidden_size_},
        hidden_states->dtype(),
        hidden_states->device());
    infinicore::op::deepseek_v4::hc_head_(
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
    if (!embed_tokens_) {
        throw std::runtime_error("DeepseekV4Model::embed_tokens called outside the first PP stage");
    }
    return embed_tokens_->forward(input_ids);
}

void DeepseekV4Model::send_hidden_states(const infinicore::Tensor &hidden_states) const {
    if (hidden_states->ndim() != 3 || hidden_states->size(1) != hc_mult_
        || hidden_states->size(2) != hidden_size_) {
        throw std::runtime_error("DeepseekV4Model: invalid PP hidden-state shape");
    }
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t shard_hidden = hidden_size_ / tp_size_;
    const size_t outer = hidden_states->size(0) * hc_mult_;
    auto local = infinicore::Tensor::empty({outer, shard_hidden}, dtype_, device_);
    local->copy_from(hidden_states
                         ->view({outer, hidden_size_})
                         ->narrow({{1, tp_rank_ * shard_hidden, shard_hidden}}));
    infinicore::op::distributed::send(
        local,
        static_cast<int>((pp_stage_ + 1) * tp_size_ + tp_rank_),
        rank_info.world_comm);
}

infinicore::Tensor DeepseekV4Model::recv_hidden_states(size_t token_count) const {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t shard_hidden = hidden_size_ / tp_size_;
    const size_t outer = token_count * hc_mult_;
    auto local = infinicore::op::distributed::recv(
        {outer, shard_hidden},
        dtype_,
        device_,
        static_cast<int>((pp_stage_ - 1) * tp_size_ + tp_rank_),
        rank_info.world_comm);
    if (tp_size_ == 1) {
        return local->view({token_count, hc_mult_, hidden_size_});
    }
    auto gathered = infinicore::op::distributed::allgather(local, tp_size_, rank_info.comm);
    return gathered->view({tp_size_, outer, shard_hidden})
        ->permute({1, 0, 2})
        ->contiguous()
        ->view({outer, hidden_size_})
        ->view({token_count, hc_mult_, hidden_size_});
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
