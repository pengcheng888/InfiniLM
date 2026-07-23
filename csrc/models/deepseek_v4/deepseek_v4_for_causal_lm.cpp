#include "deepseek_v4_for_causal_lm.hpp"

#include "../../cache/kv_cache.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {

DeepseekV4ForCausalLM::DeepseekV4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                             const infinicore::Device &device) {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto dtype = model_config->get_dtype();

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}


void DeepseekV4ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.kv_cache_vec.clear();
    forward_context.deepseek_v4_kv_cache_vec.clear();

    if (cache_config == nullptr) {
        cache_config_.reset();
        return;
    }

    auto paged_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
    if (paged_cache_config == nullptr) {
        throw std::runtime_error("DeepseekV4ForCausalLM::reset_cache requires paged KV cache");
    }

    cache_config_ = cache_config->unique_copy();
    const auto device = infinilm::global_state::get_tensor_model_parallel_rank_info().device;
    const auto dtype = model_config_->get_dtype();
    const size_t num_blocks = paged_cache_config->num_blocks();
    const size_t num_hidden_layers = model_config_->get<size_t>("num_hidden_layers");
    const size_t head_dim = model_config_->get<size_t>("head_dim");
    const size_t index_head_dim = model_config_->get_or<size_t>("index_head_dim", 128);
    const auto compress_ratios = model_config_->get<std::vector<size_t>>("compress_ratios");
    const size_t swa_num_blocks = std::max<size_t>(1, num_blocks / 10);
    const size_t c4_page_size = 64;
    const size_t index_quant_block_size = 128;
    if (index_head_dim % index_quant_block_size != 0) {
        throw std::runtime_error("DeepseekV4ForCausalLM::reset_cache requires index_head_dim divisible by 128");
    }
    const size_t indexer_page_bytes = c4_page_size * (index_head_dim + (index_head_dim / index_quant_block_size) * sizeof(float));

    forward_context.kv_cache_vec.reserve(num_hidden_layers);
    forward_context.deepseek_v4_kv_cache_vec.reserve(num_hidden_layers);
    for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
        const size_t compress_ratio = layer_idx < compress_ratios.size() ? compress_ratios[layer_idx] : 0;
        infinilm::global_state::DeepSeekV4LayerKVCache layer_cache;
        layer_cache.swa_cache_raw = infinicore::Tensor::zeros({swa_num_blocks, 149760}, infinicore::DataType::U8, device);
        layer_cache.c4_cache_raw = infinicore::Tensor::zeros({num_blocks, 37440}, infinicore::DataType::U8, device);
        if (compress_ratio == 4) {
            layer_cache.c4_indexer_cache_raw = infinicore::Tensor::zeros({num_blocks, indexer_page_bytes}, infinicore::DataType::U8, device);
        }
        layer_cache.c128_cache_raw = infinicore::Tensor::zeros({num_blocks, 1728}, infinicore::DataType::U8, device);
        layer_cache.kv_scale = infinicore::Tensor::ones({1}, infinicore::DataType::F32, device);

        if (compress_ratio != 0) {
            const size_t coeff = compress_ratio == 4 ? 2 : 1;
            const size_t ring_size = compress_ratio == 4 ? 8 : 128;
            size_t state_rows = swa_num_blocks * ring_size + ring_size + 1;
            state_rows = ((state_rows + compress_ratio - 1) / compress_ratio) * compress_ratio;
            layer_cache.compressor_state = infinicore::Tensor::zeros({state_rows, 2 * coeff * head_dim}, infinicore::DataType::F32, device);
            if (compress_ratio == 4) {
                layer_cache.indexer_compressor_state = infinicore::Tensor::zeros({state_rows, 2 * coeff * index_head_dim}, infinicore::DataType::F32, device);
            }
        }

        // Expose the primary SWA cache through the legacy getter while keeping
        // the full DSv4 cache tuple in the dedicated forward context field.
        forward_context.kv_cache_vec.push_back(layer_cache.swa_cache_raw);
        forward_context.deepseek_v4_kv_cache_vec.push_back(std::move(layer_cache));
    }
    infinicore::context::syncStream();
}

infinilm::InfinilmModel::Output DeepseekV4ForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    if (logits->ndim() == 2) {
        logits = logits->view({1, logits->size(0), logits->size(1)});
    }
    return {logits, hidden_states};
}

infinicore::Tensor DeepseekV4ForCausalLM::logits_from_hidden(const infinicore::Tensor &hidden_states) const {
    auto mutable_hidden = hidden_states;
    auto logits = lm_head_->forward(mutable_hidden);
    if (logits->ndim() == 2) {
        logits = logits->view({1, logits->size(0), logits->size(1)});
    }
    return logits;
}

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v4_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("deepseek_v4" != model_type) {
        throw std::runtime_error("infinilm::models::deepseek_v4::create_deepseek_v4_model_config: model_type is not deepseek_v4");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if ((!config_json.contains("quantization_config") || config_json["quantization_config"].is_null()) && config_json.contains("compression_config")) {
        config_json["quantization_config"] = config_json["compression_config"];
    }
    if (config_json.contains("quantization_config") && config_json["quantization_config"].is_object()) {
        config_json["quantization_config"]["quant_method"] = "compressed-tensors";
    }
    if (!config_json.contains("qk_nope_head_dim")) {
        config_json["qk_nope_head_dim"] = config_json.value("head_dim", 512) - config_json.value("qk_rope_head_dim", 64);
    }
    return std::make_shared<infinilm::config::ModelConfig>(config_json);
}

} // namespace infinilm::models::deepseek_v4

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    deepseek_v4,
    infinilm::models::deepseek_v4::DeepseekV4ForCausalLM,
    infinilm::models::deepseek_v4::create_deepseek_v4_model_config);
} // namespace
