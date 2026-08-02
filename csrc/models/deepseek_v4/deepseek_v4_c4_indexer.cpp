#include "deepseek_v4_c4_indexer.hpp"

#include "../../global_state/forward_context.hpp"
#include "deepseek_v4_profile.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_cache.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"
#include "infinicore/ops/deepseek_v4_fused_rope.hpp"
#include "infinicore/ops/deepseek_v4_sparse_attn_indexer.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
namespace {

constexpr size_t kDsv4C4PageSize = 64;

} // namespace

DeepseekV4C4Indexer::DeepseekV4C4Indexer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    const auto quantization_method = model_config->get_quantization_method();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t q_lora_rank = model_config->get<size_t>("q_lora_rank");
    index_n_heads_ = model_config->get<size_t>("index_n_heads");
    index_head_dim_ = model_config->get<size_t>("index_head_dim");
    weight_scale_ = static_cast<float>(std::pow(static_cast<double>(index_head_dim_), -0.5) * std::pow(static_cast<double>(index_n_heads_), -0.5));
    const size_t compressor_proj_size = 2 * index_head_dim_;
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_PARAMETER_INIT(ape, ({kDsv4C4PageSize / 16, compressor_proj_size}, infinicore::DataType::F32, device));
    wgate_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wgate", hidden_size, compressor_proj_size, false, dtype, device);
    wkv_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wkv", hidden_size, compressor_proj_size, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, index_head_dim_, rms_norm_eps, dtype, device);
    wq_b_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wq_b", q_lora_rank, index_n_heads_ * index_head_dim_, quantization_method, false, dtype, device);
    weights_proj_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "weights_proj", hidden_size, index_n_heads_, false, dtype, device);
}

void DeepseekV4C4Indexer::process_weights_after_loading() {
    wgate_->process_weights_after_loading();
    wkv_->process_weights_after_loading();
    wq_b_->process_weights_after_loading();
    weights_proj_->process_weights_after_loading();
}

void DeepseekV4C4Indexer::reset_runtime_state() const {
    wgate_->reset_runtime_state();
    wkv_->reset_runtime_state();
    wq_b_->reset_runtime_state();
    weights_proj_->reset_runtime_state();
}

infinicore::Tensor DeepseekV4C4Indexer::compute_q(const infinicore::Tensor &q_lora, size_t seq_len) const {
    auto q_in = q_lora;
    return wq_b_->forward(q_in)->view({seq_len, index_n_heads_, index_head_dim_});
}

infinicore::Tensor DeepseekV4C4Indexer::compute_weights(const infinicore::Tensor &hidden_states) const {
    auto x = hidden_states;
    return weights_proj_->forward(x);
}

infinicore::Tensor DeepseekV4C4Indexer::forward_kv_score(const infinicore::Tensor &hidden_states) const {
    auto x0 = hidden_states;
    auto kv = wkv_->forward(x0);
    auto x1 = hidden_states;
    auto gate = wgate_->forward(x1);
    return infinicore::op::cat({kv, gate}, -1);
}

void DeepseekV4C4Indexer::forward(
    const infinicore::Tensor &hidden_states,
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
    const infinicore::Tensor &c4_sparse_indices) const {
    {
        profile::ScopedTimer timer(profile::Event::AttentionC4IndexerCompress, seq_len);

        auto indexer_kv_score = forward_kv_score(hidden_states);
        if (indexer_kv_score->ndim() != 2 || indexer_kv_score->size(1) != 4 * index_head_dim_) {
            throw std::runtime_error("DeepseekV4C4Indexer::forward C4 indexer compressor output shape mismatch");
        }

        auto indexer_kv = infinicore::op::deepseek_v4_c4_compress_stateful(indexer_kv_score,
                                                                           ape_,
                                                                           indexer_compressor_state,
                                                                           c4_write_loc,
                                                                           c4_extra_loc,
                                                                           pos_ids);
        infinicore::op::deepseek_v4_compress_fused_norm_rope_(indexer_kv,
                                                              norm_->weight(),
                                                              norm_->eps(),
                                                              rope_freqs_cis,
                                                              c4_positions);

        infinicore::op::deepseek_v4_indexer_rotate_(indexer_kv, true);

        infinicore::op::deepseek_v4_store_indexer_raw_cache_(indexer_kv,
                                                             c4_indexer_cache_raw,
                                                             c4_out_loc,
                                                             static_cast<int>(kDsv4C4PageSize));
    }

    infinicore::Tensor indexer_q;
    infinicore::Tensor indexer_weights;
    {
        profile::ScopedTimer timer(profile::Event::AttentionC4IndexerQuery, seq_len);
        indexer_q = compute_q(q_lora, seq_len);
        auto indexer_q_rope = indexer_q->narrow({{2, index_head_dim_ - qk_rope_head_dim, qk_rope_head_dim}});
        infinicore::op::deepseek_v4_fused_rope_(indexer_q_rope, std::nullopt, rope_freqs_cis, pos_ids, false);
        infinicore::op::deepseek_v4_indexer_rotate_(indexer_q, true);
        indexer_weights = compute_weights(hidden_states);
    }
    {
        profile::ScopedTimer timer(profile::Event::AttentionC4IndexerSparse, seq_len);
        const auto max_c4_seq_len = static_cast<int>(page_table->size(1) * kDsv4C4PageSize);
        infinicore::op::deepseek_v4_c4_sparse_attn_indexer_no_logits_(indexer_q,
                                                                      indexer_weights,
                                                                      c4_indexer_cache_raw,
                                                                      c4_topk_lengths_raw,
                                                                      page_table,
                                                                      c4_sparse_indices,
                                                                      max_c4_seq_len,
                                                                      static_cast<int>(kDsv4C4PageSize),
                                                                      weight_scale_,
                                                                      false);
    }
}

} // namespace infinilm::models::deepseek_v4
