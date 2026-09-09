#include "deepseek_v4_gate.hpp"

#include "infinicore/ops/deepseek_v4/biased_topk.hpp"
#include "infinicore/ops/deepseek_v4/hash_topk.hpp"
#include "infinicore/ops/deepseek_v4/linear_bf16_fp32.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v4 {

DeepseekV4MoEGate::DeepseekV4MoEGate(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    num_experts_ = model_config->get_or_alias<size_t>("n_routed_experts", "num_experts", 0);
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    norm_topk_prob_ = model_config->get_or<bool>("norm_topk_prob", true);
    scoring_func_ = model_config->get_or<std::string>("scoring_func", "sqrtsoftplus");
    is_hash_ = layer_idx < model_config->get_or<size_t>("num_hash_layers", 0);

    if (num_experts_ == 0) {
        throw std::runtime_error("DeepseekV4MoEGate: n_routed_experts/num_experts is required");
    }
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts_, hidden_size}, model_config->get_dtype(), device));
    if (is_hash_) {
        const size_t vocab_size = model_config->get<size_t>("vocab_size");
        INFINICORE_NN_PARAMETER_INIT(tid2eid, ({vocab_size, num_experts_per_tok_}, infinicore::DataType::I64, device));
    } else {
        INFINICORE_NN_PARAMETER_INIT(bias, ({num_experts_}, infinicore::DataType::F32, device));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> DeepseekV4MoEGate::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &input_ids) const {
    if (hidden_states->ndim() != 2) {
        throw std::runtime_error("DeepseekV4MoEGate::forward expects hidden_states [tokens, hidden]");
    }
    if (input_ids->ndim() != 1 || input_ids->size(0) != hidden_states->size(0)) {
        throw std::runtime_error("DeepseekV4MoEGate::forward expects flat input_ids [tokens]");
    }

    auto router_logits = infinicore::op::deepseek_v4::linear_bf16_fp32(hidden_states, weight_);
    auto router_scores = infinicore::Tensor::empty(
        {hidden_states->size(0), num_experts_per_tok_},
        infinicore::DataType::F32,
        hidden_states->device());
    auto router_indices = infinicore::Tensor::empty(
        {hidden_states->size(0), num_experts_per_tok_},
        infinicore::DataType::I32,
        hidden_states->device());

    if (is_hash_) {
        infinicore::op::deepseek_v4::hash_topk_(
            router_scores,
            router_indices,
            router_logits,
            input_ids,
            tid2eid_,
            0,
            1.0f,
            scoring_func_);
    } else {
        infinicore::op::deepseek_v4::topk_(
            router_scores,
            router_indices,
            router_logits,
            bias_,
            norm_topk_prob_,
            scoring_func_);
    }

    return {router_scores, router_indices};
}

} // namespace infinilm::models::deepseek_v4
