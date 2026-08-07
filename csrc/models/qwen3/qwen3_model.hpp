#pragma once

#include "../../models/infinilm_model.hpp"
#include "qwen3_decoder_layer.hpp"
#include "qwen3_rms_norm.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::qwen3 {

class Qwen3Model : public infinicore::nn::Module {
public:
    Qwen3Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
               const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;
    infinicore::Tensor forward_naive(const infinilm::InfinilmModel::Input &input) const;
    infinicore::Tensor embed_tokens(const infinicore::Tensor &input_ids) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(Qwen3DecoderLayer, layers);
    INFINICORE_NN_MODULE(Qwen3RMSNorm, norm);
};

} // namespace infinilm::models::qwen3
