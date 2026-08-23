#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "deepseek_v4_compressor.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4C4Indexer : public infinicore::nn::Module {
public:
    DeepseekV4C4Indexer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        const infinicore::Device &device);

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

    infinicore::Tensor compute_q(const infinicore::Tensor &q_lora,
                                 const infinicore::Tensor &pos_ids,
                                 size_t seq_len,
                                 const infinicore::Tensor &rope_freqs_cis,
                                 size_t qk_rope_head_dim) const;
    infinicore::Tensor compute_weights(const infinicore::Tensor &hidden_states) const;
    void forward(const infinicore::Tensor &hidden_states,
                 const infinicore::Tensor &q_lora,
                 const infinicore::Tensor &pos_ids,
                 size_t seq_len,
                 const infinicore::Tensor &rope_freqs_cis,
                 size_t qk_rope_head_dim,
                 const infinicore::Tensor &indexer_compressor_state,
                 const infinicore::Tensor &c4_indexer_cache_raw,
                 const infinicore::Tensor &c4_out_loc,
                 const infinicore::Tensor &c4_positions,
                 const infinicore::Tensor &c4_write_loc,
                 const infinicore::Tensor &c4_extra_loc,
                 const infinicore::Tensor &c4_topk_lengths_raw,
                 const infinicore::Tensor &page_table,
                 const infinicore::Tensor &c4_sparse_indices) const;
    float weight_scale() const { return weight_scale_; }
    size_t index_n_heads() const { return index_n_heads_; }
    size_t index_head_dim() const { return index_head_dim_; }

private:
    INFINICORE_NN_MODULE(DeepseekV4Compressor, compressor);
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wq_b_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> weights_proj_;
    size_t index_n_heads_{0};
    size_t index_head_dim_{0};
    float weight_scale_{1.0f};
};

} // namespace infinilm::models::deepseek_v4
