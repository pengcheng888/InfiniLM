#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "deepseek_v4_rms_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <optional>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Compressor : public infinicore::nn::Module {
public:
    DeepseekV4Compressor(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t head_dim,
                         size_t compress_ratio,
                         const infinicore::Device &device);

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

    void forward(const infinicore::Tensor &hidden_states,
                 const infinicore::Tensor &pos_ids,
                 size_t seq_len,
                 const infinicore::Tensor &rope_freqs_cis,
                 const infinicore::Tensor &compressor_state,
                 const infinicore::Tensor &cache_raw,
                 const infinicore::Tensor &out_loc,
                 const infinicore::Tensor &compress_positions,
                 const infinicore::Tensor &write_loc,
                 std::optional<infinicore::Tensor> extra_loc) const;

private:
    infinicore::Tensor forward_kv_score(const infinicore::Tensor &hidden_states) const;

    INFINICORE_NN_PARAMETER(ape);
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wgate_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wkv_;
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, norm);
    size_t head_dim_{0};
    size_t compress_ratio_{0};
    size_t page_size_{0};
    size_t proj_size_{0};
};

} // namespace infinilm::models::deepseek_v4
