#include "deepseek_v4_attention.hpp"

#include "../../global_state/global_state.hpp"

#include <stdexcept>
#include <vector>

namespace infinilm::models::deepseek_v4 {

DeepseekV4Compressor::DeepseekV4Compressor(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t compress_ratio,
                                           size_t compressor_head_dim,
                                           const infinicore::Device &device) {
    if (compress_ratio == 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4Compressor: compress_ratio must be non-zero");
    }
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t coeff = compress_ratio == 4 ? 2 : 1;
    const size_t proj_size = coeff * compressor_head_dim;
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_PARAMETER_INIT(ape, ({compress_ratio, proj_size}, infinicore::DataType::F32, device));
    wgate_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wgate", hidden_size, proj_size, false, dtype, device);
    wkv_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wkv", hidden_size, proj_size, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, compressor_head_dim, rms_norm_eps, dtype, device);
}

void DeepseekV4Compressor::process_weights_after_loading() {
    wgate_->process_weights_after_loading();
    wkv_->process_weights_after_loading();
}

void DeepseekV4Compressor::reset_runtime_state() const {
    wgate_->reset_runtime_state();
    wkv_->reset_runtime_state();
}

DeepseekV4C4Indexer::DeepseekV4C4Indexer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    const auto quantization_method = model_config->get_quantization_method();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t q_lora_rank = model_config->get<size_t>("q_lora_rank");
    const size_t index_n_heads = model_config->get<size_t>("index_n_heads");
    const size_t index_head_dim = model_config->get<size_t>("index_head_dim");

    wq_b_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wq_b", q_lora_rank, index_n_heads * index_head_dim, quantization_method, false, dtype, device);
    weights_proj_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "weights_proj", hidden_size, index_n_heads, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(compressor, model_config, 4, index_head_dim, device);
}

void DeepseekV4C4Indexer::process_weights_after_loading() {
    wq_b_->process_weights_after_loading();
    weights_proj_->process_weights_after_loading();
    compressor_->process_weights_after_loading();
}

void DeepseekV4C4Indexer::reset_runtime_state() const {
    wq_b_->reset_runtime_state();
    weights_proj_->reset_runtime_state();
    compressor_->reset_runtime_state();
}

DeepseekV4Attention::DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         size_t layer_idx,
                                         const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto dtype = model_config->get_dtype();
    const auto quantization_method = model_config->get_quantization_method();
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);

    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    num_attention_heads_ = model_config->get<size_t>("num_attention_heads");
    num_key_value_heads_ = model_config->get<size_t>("num_key_value_heads");
    q_lora_rank_ = model_config->get<size_t>("q_lora_rank");
    o_lora_rank_ = model_config->get<size_t>("o_lora_rank");
    qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    o_groups_ = model_config->get<size_t>("o_groups");
    rms_norm_eps_ = model_config->get<double>("rms_norm_eps");

    if (num_key_value_heads_ != 1) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4Attention: num_key_value_heads must be 1");
    }
    if (num_attention_heads_ % tp_size != 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4Attention: num_attention_heads must be divisible by tp_size");
    }
    if (o_groups_ % tp_size != 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4Attention: o_groups must be divisible by tp_size");
    }

    INFINICORE_NN_PARAMETER_INIT(attn_sink, ({num_attention_heads_}, infinicore::DataType::F32, device));

    wq_a_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wq_a", hidden_size_, q_lora_rank_, quantization_method, false, dtype, device);
    wkv_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
        "wkv", hidden_size_, head_dim_, quantization_method, false, dtype, device);
    wq_b_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "wq_b",
        q_lora_rank_,
        num_attention_heads_ * head_dim_,
        quantization_method,
        false,
        dtype,
        device,
        tp_rank,
        tp_size);

    INFINICORE_NN_MODULE_INIT(q_norm, q_lora_rank_, rms_norm_eps_, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_norm, head_dim_, rms_norm_eps_, dtype, device);

    wq_b_post_norm_ = std::make_shared<DeepseekV4RMSNorm>(head_dim_, rms_norm_eps_, dtype, device);

    wo_a_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
        "wo_a",
        num_attention_heads_ * head_dim_ / o_groups_,
        o_groups_ * o_lora_rank_,
        false,
        dtype,
        device,
        tp_rank,
        tp_size);
    wo_b_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "wo_b",
        o_groups_ * o_lora_rank_,
        hidden_size_,
        quantization_method,
        false,
        dtype,
        device,
        tp_rank,
        tp_size,
        rank_info.comm);

    const auto compress_ratios = model_config->get<std::vector<size_t>>("compress_ratios");
    const size_t compress_ratio = layer_idx_ < compress_ratios.size() ? compress_ratios[layer_idx_] : 0;
    if (compress_ratio != 0) {
        INFINICORE_NN_MODULE_INIT(compressor, model_config, compress_ratio, head_dim_, device);
        if (compress_ratio == 4) {
            INFINICORE_NN_MODULE_INIT(indexer, model_config, device);
        }
    }
}

infinicore::Tensor DeepseekV4Attention::forward(const infinicore::Tensor &, const infinicore::Tensor &) const {
    throw std::runtime_error("infinilm::models::deepseek_v4::DeepseekV4Attention::forward is not implemented yet; weight loading only");
}

} // namespace infinilm::models::deepseek_v4
