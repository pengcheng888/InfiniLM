#pragma once

#include "../../config/model_config.hpp"
#include "qwen3_attention.hpp"
#include "qwen3_mlp.hpp"
#include "qwen3_rms_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <tuple>

namespace infinilm::models::qwen3 {

class Qwen3DecoderLayer : public infinicore::nn::Module {
public:
    Qwen3DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t layer_idx,
                      const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

private:
    INFINICORE_NN_MODULE(Qwen3RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(Qwen3RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(Qwen3Attention, self_attn);
    INFINICORE_NN_MODULE(Qwen3MLP, mlp);
    size_t layer_idx_;
};

} // namespace infinilm::models::qwen3
