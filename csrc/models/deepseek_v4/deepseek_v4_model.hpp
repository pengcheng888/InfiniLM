#pragma once

#include "../../models/infinilm_model.hpp"
#include "deepseek_v4_decoder_layer.hpp"
#include "deepseek_v4_scratch.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
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
    bool is_first_pp_stage() const { return pp_stage_ == 0; }
    bool is_last_pp_stage() const { return pp_stage_ + 1 == pp_size_; }

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    void send_hidden_states(const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor recv_hidden_states(size_t token_count) const;

    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(DeepseekV4DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    INFINICORE_NN_PARAMETER(hc_head_fn);
    INFINICORE_NN_PARAMETER(hc_head_base);
    INFINICORE_NN_PARAMETER(hc_head_scale);

    size_t hidden_size_{0};
    size_t hc_mult_{4};
    infinicore::DataType dtype_{infinicore::DataType::BF16};
    size_t pp_size_{1};
    size_t pp_stage_{0};
    size_t tp_size_{1};
    size_t tp_rank_{0};
    infinicore::Device device_;
    double rms_norm_eps_{1e-6};
    double hc_eps_{1e-6};

    mutable DeepseekV4FlatScratchBuffer embedding_hc_expand_scratch_;
    mutable DeepseekV4FlatScratchBuffer hc_head_collapse_scratch_;
};

} // namespace infinilm::models::deepseek_v4
