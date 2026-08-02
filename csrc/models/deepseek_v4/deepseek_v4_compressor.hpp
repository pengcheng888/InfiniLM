#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "deepseek_v4_rms_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4CSACompressor : public infinicore::nn::Module {
public:
    DeepseekV4CSACompressor(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t head_dim,
                            const infinicore::Device &device);

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

    void forward(const infinicore::Tensor &hidden_states,
                 const infinicore::Tensor &pos_ids,
                 size_t seq_len,
                 const infinicore::Tensor &rope_freqs_cis,
                 const infinicore::Tensor &compressor_state,
                 const infinicore::Tensor &c4_cache_raw,
                 const infinicore::Tensor &c4_out_loc,
                 const infinicore::Tensor &c4_positions,
                 const infinicore::Tensor &c4_write_loc,
                 const infinicore::Tensor &c4_extra_loc) const;

private:
    infinicore::Tensor forward_kv_score(const infinicore::Tensor &hidden_states) const;

    INFINICORE_NN_PARAMETER(ape);
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wgate_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wkv_;
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, norm);
    size_t head_dim_{0};
};

class DeepseekV4HCACompressor : public infinicore::nn::Module {
public:
    DeepseekV4HCACompressor(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t head_dim,
                            const infinicore::Device &device);

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

    void forward(const infinicore::Tensor &hidden_states,
                 const infinicore::Tensor &pos_ids,
                 size_t seq_len,
                 const infinicore::Tensor &rope_freqs_cis,
                 const infinicore::Tensor &compressor_state,
                 const infinicore::Tensor &c128_cache_raw,
                 const infinicore::Tensor &c128_out_loc,
                 const infinicore::Tensor &c128_positions,
                 const infinicore::Tensor &c128_write_loc) const;

private:
    infinicore::Tensor forward_kv_score(const infinicore::Tensor &hidden_states) const;

    INFINICORE_NN_PARAMETER(ape);
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wgate_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wkv_;
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, norm);
    size_t head_dim_{0};
};

} // namespace infinilm::models::deepseek_v4
