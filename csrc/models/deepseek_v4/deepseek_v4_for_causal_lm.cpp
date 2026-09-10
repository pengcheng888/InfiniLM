#include "deepseek_v4_for_causal_lm.hpp"

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/nn/rope.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::deepseek_v4 {
namespace {

constexpr size_t kDsv4FlashMlaBytesPerToken = 584;

size_t round_up_to_multiple(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

size_t flashmla_raw_cache_page_bytes(size_t page_size) {
    return round_up_to_multiple(kDsv4FlashMlaBytesPerToken * page_size, 576);
}

} // namespace

DeepseekV4ForCausalLM::DeepseekV4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                             const infinicore::Device &device)
    : device_(device) {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto dtype = model_config->get_dtype();

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
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
    auto mutable_hidden_states = hidden_states;
    auto logits = lm_head_->forward(mutable_hidden_states);
    if (logits->ndim() == 2) {
        logits = logits->view({1, logits->size(0), logits->size(1)});
    }
    return logits;
}

void DeepseekV4ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &kv_cache_vec = forward_context.kv_cache_vec;
    kv_cache_vec.clear();

    if (cache_config == nullptr) {
        cache_config_.reset();
        forward_context.sched_meta.clear();
        return;
    }
    const auto attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    if (attention_backend != backends::AttentionBackend::PAGED_ATTN
        && attention_backend != backends::AttentionBackend::FLASH_ATTN) {
        throw std::runtime_error("DeepseekV4ForCausalLM requires paged attention cache");
    }
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
    if (paged_config == nullptr) {
        throw std::runtime_error("DeepseekV4ForCausalLM requires PagedKVCacheConfig");
    }
    cache_config_ = cache_config->unique_copy();

    const size_t num_hidden_layers = model_config_->get<size_t>("num_hidden_layers");
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t pp_size = static_cast<size_t>(rank_info.pp_size);
    const size_t pp_stage = static_cast<size_t>(rank_info.pp_stage);
    const size_t local_layer_begin = num_hidden_layers * pp_stage / pp_size;
    const size_t local_layer_end = num_hidden_layers * (pp_stage + 1) / pp_size;

    kv_cache_vec.resize(num_hidden_layers);
    const size_t page_size = paged_config->block_size();
    for (size_t layer_idx = local_layer_begin; layer_idx < local_layer_end; ++layer_idx) {
        kv_cache_vec[layer_idx] = infinicore::Tensor::zeros(
            {paged_config->num_blocks(), flashmla_raw_cache_page_bytes(page_size)},
            infinicore::DataType::U8,
            device_);
    }
    infinicore::context::syncStream();
    forward_context.sched_meta.resize_flash_mla_sched_meta(1);
}

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if (model_type != "deepseek_v4") {
        throw std::runtime_error("create_deepseek_v4_model_config: model_type is not deepseek_v4");
    }

    auto &config_json = model_config->get_config_json();
    if ((!config_json.contains("quantization_config") || config_json["quantization_config"].is_null())
        && config_json.contains("compression_config")) {
        config_json["quantization_config"] = config_json["compression_config"];
    }
    if (config_json.contains("quantization_config") && config_json["quantization_config"].is_object()) {
        config_json["quantization_config"]["quant_method"] = "compressed-tensors";
    }
    if (!config_json.contains("qk_nope_head_dim")) {
        config_json["qk_nope_head_dim"] = config_json.value("head_dim", 512) - config_json.value("qk_rope_head_dim", 64);
    }
    if (!config_json.contains("intermediate_size") && config_json.contains("moe_intermediate_size")) {
        config_json["intermediate_size"] = config_json["moe_intermediate_size"];
    }
    if (!config_json.contains("num_experts") && config_json.contains("n_routed_experts")) {
        config_json["num_experts"] = config_json["n_routed_experts"];
    }
    if (!config_json.contains("moe_router_backend")) {
        config_json["moe_router_backend"] = "sigmoid";
    }
    if (!config_json.contains("e_score_correction_bias")) {
        config_json["e_score_correction_bias"] = false;
    }
    if (!config_json.contains("torch_dtype") && !config_json.contains("dtype")) {
        config_json["torch_dtype"] = "bfloat16";
    }
    const size_t sliding_window = model_config->get<size_t>("sliding_window");
    if (sliding_window == 0) {
        throw std::invalid_argument("DeepSeek-V4 sliding_window must be positive");
    }
    config_json["swa_topk"] = sliding_window;
    auto fixed_config = std::make_shared<infinilm::config::ModelConfig>(config_json);
    fixed_config->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_J);
    return fixed_config;
}

} // namespace infinilm::models::deepseek_v4

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    deepseek_v4,
    infinilm::models::deepseek_v4::DeepseekV4ForCausalLM,
    infinilm::models::deepseek_v4::create_deepseek_v4_model_config);
} // namespace
