#include "deepseek_v4_gate.hpp"

#include "deepseek_v4_profile.hpp"

#include "infinicore/ops/deepseek_v4_biased_topk.hpp"
#include "infinicore/ops/deepseek_v4_hash_topk.hpp"
#include "infinicore/ops/deepseek_v4_linear_bf16_fp32.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {

DeepseekV4MoEGate::DeepseekV4MoEGate(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t layer_idx,
                                     const infinicore::Device &device) {
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_experts = model_config->get_or_alias<size_t>("n_routed_experts", "num_experts", 0);
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t num_experts_per_tok = model_config->get<size_t>("num_experts_per_tok");
    const size_t num_hash_layers = model_config->get_or<size_t>("num_hash_layers", 0);
    num_experts_per_tok_ = num_experts_per_tok;
    num_experts_ = num_experts;
    norm_topk_prob_ = model_config->get_or<bool>("norm_topk_prob", true);
    scoring_func_ = model_config->get_or<std::string>("scoring_func", "sqrtsoftplus");
    is_hash_ = layer_idx < num_hash_layers;
    if (num_experts == 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4MoEGate: n_routed_experts is required");
    }
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts, hidden_size}, model_config->get_dtype(), device));
    if (is_hash_) {
        INFINICORE_NN_PARAMETER_INIT(tid2eid, ({vocab_size, num_experts_per_tok}, infinicore::DataType::I64, device));
    } else {
        INFINICORE_NN_PARAMETER_INIT(bias, ({num_experts}, infinicore::DataType::F32, device));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
DeepseekV4MoEGate::forward(const infinicore::Tensor &hidden_states,
                           const infinicore::Tensor &input_ids) const {
    if (hidden_states->ndim() != 2) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4MoEGate::forward expects hidden_states [tokens, hidden]");
    }
    const size_t token_count = hidden_states->size(0);
    infinicore::Tensor router_logits;
    {
        profile::ScopedTimer timer(profile::Event::MoeGate, token_count);
        router_logits = router_logits_scratch_.get({hidden_states->size(0), num_experts_},
                                                   infinicore::DataType::F32,
                                                   hidden_states->device());

        infinicore::op::deepseek_v4_linear_bf16_fp32_(router_logits, // [ntoken, 256]
                                                      hidden_states,
                                                      weight_);
    }
    auto router_scores = router_scores_scratch_.get({hidden_states->size(0), num_experts_per_tok_},
                                                    infinicore::DataType::F32,
                                                    hidden_states->device());

    auto router_indices = router_indices_scratch_.get({hidden_states->size(0), num_experts_per_tok_},
                                                      infinicore::DataType::I32,
                                                      hidden_states->device());
    {
        profile::ScopedTimer timer(profile::Event::MoeTopk, token_count);
        if (is_hash_) {
            infinicore::op::deepseek_v4_hash_topk_(
                router_scores,  // [ntoken, 6]
                router_indices, // [ntoken, 6]
                router_logits,
                input_ids,
                tid2eid_,
                0,
                1.0f,
                scoring_func_);
        } else {
            infinicore::op::deepseek_v4_topk_(
                router_scores,  // [ntoken, 6]
                router_indices, // [ntoken, 6]
                router_logits,
                bias_,
                norm_topk_prob_);
        }
    }
    return {router_scores, router_indices};
}

} // namespace infinilm::models::deepseek_v4
