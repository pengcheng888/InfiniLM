#include "deepseek_v4_c4_indexer.hpp"

#include "../../global_state/forward_context.hpp"
#include "deepseek_v4_profile.hpp"
#include "infinicore/ops/deepseek_v4_c4_paged_mqa_logits.hpp"
#include "infinicore/ops/deepseek_v4_c4_paged_mqa_with_topk_transform_512.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_cache.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"
#include "infinicore/ops/deepseek_v4_fused_q_indexer_rope_hadamard_quant.hpp"
#include "infinicore/ops/deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang.hpp"
#include "infinicore/ops/deepseek_v4_fused_rope.hpp"
#include "infinicore/ops/deepseek_v4_sparse_attn_indexer.hpp"
#include "infinicore/ops/deepseek_v4_topk_transform_512.hpp"
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
    INFINICORE_NN_MODULE_INIT(compressor,
                              model_config,
                              index_head_dim_,
                              static_cast<size_t>(4),
                              device,
                              DeepseekV4CompressorStoreKind::Indexer);
    wq_b_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wq_b", q_lora_rank, index_n_heads_ * index_head_dim_, quantization_method, false, dtype, device);
    weights_proj_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "weights_proj", hidden_size, index_n_heads_, false, dtype, device);
}

void DeepseekV4C4Indexer::process_weights_after_loading() {
    compressor_->process_weights_after_loading();
    wq_b_->process_weights_after_loading();
    weights_proj_->process_weights_after_loading();
}

void DeepseekV4C4Indexer::reset_runtime_state() const {
    compressor_->reset_runtime_state();
    wq_b_->reset_runtime_state();
    weights_proj_->reset_runtime_state();
}

infinicore::Tensor DeepseekV4C4Indexer::compute_q(const infinicore::Tensor &q_lora,
                                                  const infinicore::Tensor &pos_ids,
                                                  size_t seq_len,
                                                  const infinicore::Tensor &rope_freqs_cis,
                                                  size_t qk_rope_head_dim) const {
    auto q_in = q_lora;
    auto indexer_q = wq_b_->forward(q_in)->view({seq_len, index_n_heads_, index_head_dim_});
    auto indexer_q_rope = indexer_q->narrow({{2, index_head_dim_ - qk_rope_head_dim, qk_rope_head_dim}});
    infinicore::op::deepseek_v4_fused_rope_(indexer_q_rope, std::nullopt, rope_freqs_cis, pos_ids, false);
    infinicore::op::deepseek_v4_indexer_rotate_(indexer_q, true);
    return indexer_q;
}

infinicore::Tensor DeepseekV4C4Indexer::compute_weights(const infinicore::Tensor &hidden_states) const {
    auto x = hidden_states;
    return weights_proj_->forward(x);
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
    compressor_->forward(hidden_states,
                         pos_ids,
                         seq_len,
                         rope_freqs_cis,
                         indexer_compressor_state,
                         c4_indexer_cache_raw,
                         c4_out_loc,
                         c4_positions,
                         c4_write_loc,
                         c4_extra_loc);

    infinicore::Tensor indexer_q;
    infinicore::Tensor indexer_weights;
    {
        indexer_weights = compute_weights(hidden_states);
    }

    const int repeats = 1; // 1000
    for (int i = 0; i < repeats; ++i) {
        /*
        use_new_version = true; ms=216.634 ， （当use_sglang = true时，进一步降低到了211 ms)

        use_new_version = false; ms=230.634
        */
        bool use_new_version = true;
        if (use_new_version) {
            {
                profile::ScopedTimer timer(profile::Event::AttentionC4IndexerQuery, seq_len);
                auto q_lora_mut = q_lora;
                indexer_q = wq_b_->forward(q_lora_mut)->view({seq_len, index_n_heads_, index_head_dim_});
            }
            {
                profile::ScopedTimer timer(profile::Event::AttentionC4IndexerSparse, seq_len);
                const auto max_c4_seq_len = static_cast<int>(page_table->size(1) * kDsv4C4PageSize);

                auto q_fp8 = infinicore::Tensor::empty(indexer_q->shape(), infinicore::DataType::F8, indexer_q->device());
                infinicore::Tensor fused_weights;

                bool use_sglang = true;
                if (use_sglang) {
                    auto fused_weights_sglang = infinicore::Tensor::empty({seq_len, index_n_heads_, static_cast<size_t>(1)},
                                                                          infinicore::DataType::F32,
                                                                          indexer_weights->device());
                    infinicore::op::deepseek_v4_fused_q_indexer_rope_hadamard_quant_sglang_(indexer_q,
                                                                                            q_fp8,
                                                                                            indexer_weights,
                                                                                            fused_weights_sglang,
                                                                                            weight_scale_,
                                                                                            rope_freqs_cis,
                                                                                            pos_ids);
                    fused_weights = fused_weights_sglang->view(indexer_weights->shape());
                } else {
                    auto q_scale = infinicore::Tensor::empty({seq_len, index_n_heads_, static_cast<size_t>(1)}, infinicore::DataType::F32, indexer_q->device());
                    fused_weights = infinicore::Tensor::empty(indexer_weights->shape(), infinicore::DataType::F32, indexer_weights->device());
                    infinicore::op::deepseek_v4_fused_q_indexer_rope_hadamard_quant_(indexer_q,
                                                                                     indexer_weights,
                                                                                     q_fp8,
                                                                                     q_scale,
                                                                                     fused_weights,
                                                                                     weight_scale_,
                                                                                     rope_freqs_cis,
                                                                                     pos_ids);
                }

                auto logits = infinicore::Tensor::empty({seq_len, static_cast<size_t>(max_c4_seq_len)}, infinicore::DataType::F32, indexer_q->device());

                const int repeats = 1;
                for (int i = 0; i < repeats; ++i) {
                    // 1000  total_ms=165 // 这个优化相当于没有
                    bool use_deepseek_v4_c4_paged_mqa_logits_wit_topk_transform_512 = true;
                    if (use_deepseek_v4_c4_paged_mqa_logits_wit_topk_transform_512) {
                        infinicore::op::deepseek_v4_c4_paged_mqa_with_topk_transform_512_(q_fp8,
                                                                                          fused_weights,
                                                                                          c4_indexer_cache_raw,
                                                                                          c4_topk_lengths_raw,
                                                                                          page_table,
                                                                                          c4_sparse_indices,
                                                                                          max_c4_seq_len,
                                                                                          static_cast<int>(kDsv4C4PageSize),
                                                                                          false);
                    } else {
                        // 1000  total_ms=170.244
                        // 这个函数中有copy操作，将result拷贝到logits变量中。
                        infinicore::op::deepseek_v4_c4_paged_mqa_logits_(q_fp8,
                                                                         fused_weights,
                                                                         c4_indexer_cache_raw,
                                                                         c4_topk_lengths_raw,
                                                                         page_table,
                                                                         logits,
                                                                         max_c4_seq_len,
                                                                         static_cast<int>(kDsv4C4PageSize),
                                                                         false);

                        infinicore::op::deepseek_v4_topk_transform_512_kernel_(logits,
                                                                               c4_topk_lengths_raw,
                                                                               page_table,
                                                                               c4_sparse_indices,
                                                                               static_cast<int>(kDsv4C4PageSize));
                    }
                }
            }
        } else {
            {
                profile::ScopedTimer timer(profile::Event::AttentionC4IndexerQuery, seq_len);
                indexer_q = compute_q(q_lora, pos_ids, seq_len, rope_freqs_cis, qk_rope_head_dim); // for test 5000  //131.618ms
            }
            {
                //  for test 1000 forward_ms=164.857
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
    }
}

} // namespace infinilm::models::deepseek_v4
