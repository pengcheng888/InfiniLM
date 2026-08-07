#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v4_scratch.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <string>
#include <tuple>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4MoEGate : public infinicore::nn::Module {
public:
    DeepseekV4MoEGate(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t layer_idx,
                      const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &hidden_states,
                                                               const infinicore::Tensor &input_ids) const;

private:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(tid2eid);
    INFINICORE_NN_PARAMETER(bias);

    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
    bool norm_topk_prob_{true};
    bool is_hash_{true};
    std::string scoring_func_{"sqrtsoftplus"};
    mutable DeepseekV4FlatScratchBuffer router_logits_scratch_;
    mutable DeepseekV4FlatScratchBuffer router_scores_scratch_;
    mutable DeepseekV4FlatScratchBuffer router_indices_scratch_;
};

} // namespace infinilm::models::deepseek_v4
