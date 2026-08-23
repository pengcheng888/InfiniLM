#pragma once

#include "../../config/model_config.hpp"
#include "../../models/infinilm_model.hpp"
#include "glm4_moe_lite_decoder_layer.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"

#include <memory>

namespace infinilm::models::glm4_moe_lite {

class Glm4MoeLiteModel : public infinicore::nn::Module {
public:
    Glm4MoeLiteModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    size_t hidden_size_{0};

    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(Glm4MoeLiteDecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
};

} // namespace infinilm::models::glm4_moe_lite
