#include "glm4_moe_lite_for_causal_lm.hpp"

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include "infinicore/context/context.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::glm4_moe_lite {

Glm4MoeLiteForCausalLM::Glm4MoeLiteForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device) {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto dtype = model_config->get_dtype();

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    lm_head_ = this->register_module<infinilm::layers::lm_head::ParallelLMHead>(
        "lm_head",
        hidden_size,
        vocab_size,
        false,
        dtype,
        device,
        static_cast<infinicore::Size>(rank_info.tp_rank),
        static_cast<infinicore::Size>(rank_info.tp_size),
        rank_info.comm);
}

infinilm::InfinilmModel::Output Glm4MoeLiteForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = compute_lm_head_logits(hidden_states);
    if (logits->ndim() == 2) {
        logits = logits->view({1, logits->size(0), logits->size(1)});
    }
    return {logits, hidden_states};
}

infinicore::Tensor Glm4MoeLiteForCausalLM::compute_lm_head_logits(const infinicore::Tensor &hidden_states) const {
    auto mutable_hidden = hidden_states;
    return lm_head_->forward(mutable_hidden);
}

infinicore::Tensor Glm4MoeLiteForCausalLM::logits_from_hidden(const infinicore::Tensor &hidden_states) const {
    auto logits = compute_lm_head_logits(hidden_states);
    if (logits->ndim() == 2) {
        logits = logits->view({1, logits->size(0), logits->size(1)});
    }
    return logits;
}

void Glm4MoeLiteForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.kv_cache_vec.clear();
    forward_context.deepseek_v4_kv_cache_vec.clear();

    if (cache_config == nullptr) {
        cache_config_.reset();
        return;
    }
    const auto attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    if (attention_backend != backends::AttentionBackend::PAGED_ATTN
        && attention_backend != backends::AttentionBackend::FLASH_ATTN) {
        throw std::runtime_error("Glm4MoeLiteForCausalLM requires paged attention cache");
    }
    auto paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
    if (paged_config == nullptr) {
        throw std::runtime_error("Glm4MoeLiteForCausalLM requires PagedKVCacheConfig");
    }
    cache_config_ = cache_config->unique_copy();

    const size_t num_hidden_layers = model_config_->get<size_t>("num_hidden_layers");
    const size_t latent_dim = model_config_->get<size_t>("kv_lora_rank")
                            + model_config_->get<size_t>("qk_rope_head_dim");
    const auto dtype = model_config_->get_kv_cache_dtype();
    const infinicore::Device device = lm_head_->weight()->device();

    forward_context.kv_cache_vec.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        forward_context.kv_cache_vec.push_back(infinicore::Tensor::empty(
            {paged_config->num_blocks(), paged_config->block_size(), latent_dim},
            dtype,
            device));
    }
    infinicore::context::syncStream();
}

std::shared_ptr<infinilm::config::ModelConfig> create_glm4_moe_lite_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("glm4_moe_lite" != model_type) {
        throw std::runtime_error("create_glm4_moe_lite_model_config: model_type is not glm4_moe_lite");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    if (!config_json.contains("head_dim")) {
        config_json["head_dim"] = config_json.value("kv_lora_rank", 512)
                                + config_json.value("qk_rope_head_dim", 64);
    }
    if (!config_json.contains("num_experts") && config_json.contains("n_routed_experts")) {
        config_json["num_experts"] = config_json["n_routed_experts"];
    }
    if (!config_json.contains("moe_router_backend")) {
        config_json["moe_router_backend"] = "sigmoid";
    }
    if (!config_json.contains("e_score_correction_bias")) {
        config_json["e_score_correction_bias"] = true;
    }
    if (!config_json.contains("torch_dtype") && !config_json.contains("dtype")) {
        config_json["torch_dtype"] = "bfloat16";
    }
    return std::make_shared<infinilm::config::ModelConfig>(config_json);
}

} // namespace infinilm::models::glm4_moe_lite

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    glm4_moe_lite,
    infinilm::models::glm4_moe_lite::Glm4MoeLiteForCausalLM,
    infinilm::models::glm4_moe_lite::create_glm4_moe_lite_model_config);
} // namespace
