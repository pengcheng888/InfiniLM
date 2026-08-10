#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../global_state/forward_context.hpp"
#include "../../layers/lm_head/parallel_lm_head.hpp"
#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"
#include "deepseek_v4_model.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4ForCausalLM : public InfinilmModel {
public:
    DeepseekV4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          const infinicore::Device &device);

    void reset_cache(const cache::CacheConfig *cache_config) override;
    Output forward(const Input &input) const override;
    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(DeepseekV4Model, model);
    infinicore::Tensor _compute_lm_head_logits(const infinicore::Tensor &hidden_states) const;

    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> replicated_lm_head_;
    std::shared_ptr<infinilm::layers::lm_head::ParallelLMHead> parallel_lm_head_;
    bool use_parallellm_head_{true};
};

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v4_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

struct DeepseekV4KVCacheTensors {
    std::vector<infinicore::Tensor> kv_cache_tensors;
    std::vector<infinilm::global_state::DeepSeekV4LayerKVCache> deepseek_v4_kv_cache_tensors;
};

/** Implemented in `deepseek_v4_allocate_kv_cache_tensors.cpp`. */
DeepseekV4KVCacheTensors deepseek_v4_allocate_kv_cache_tensors(const cache::CacheConfig *cache_config,
                                                               const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
                                                               const backends::AttentionBackend &attention_backend);

} // namespace infinilm::models::deepseek_v4
