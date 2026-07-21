#pragma once

#include "../../models/infinilm_model.hpp"
#include "deepseek_v4_decoder_layer.hpp"
#include "deepseek_v4_rms_norm.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Model : public infinicore::nn::Module {
public:
    DeepseekV4Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;
    infinicore::Tensor embed_tokens(const infinicore::Tensor &input_ids) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(DeepseekV4DecoderLayer, layers);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, norm);

    INFINICORE_NN_PARAMETER(hc_head_fn);
    INFINICORE_NN_PARAMETER(hc_head_base);
    INFINICORE_NN_PARAMETER(hc_head_scale);

    size_t hidden_size_{0};
    size_t hc_mult_{4};
    double rms_norm_eps_{1e-6};
    double hc_eps_{1e-6};
};

} // namespace infinilm::models::deepseek_v4
