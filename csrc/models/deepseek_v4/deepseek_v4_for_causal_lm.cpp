#include "deepseek_v4_for_causal_lm.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "deepseek_v4_profile.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::deepseek_v4 {

DeepseekV4ForCausalLM::DeepseekV4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                             const infinicore::Device &device) {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto dtype = model_config->get_dtype();

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    use_parallellm_head_ = true;
    if (use_parallellm_head_) {
        parallel_lm_head_ = this->register_module<infinilm::layers::lm_head::ParallelLMHead>(
            "lm_head",
            hidden_size,
            vocab_size,
            false,
            dtype,
            device,
            static_cast<infinicore::Size>(rank_info.tp_rank),
            static_cast<infinicore::Size>(rank_info.tp_size),
            rank_info.comm);
    } else {
        replicated_lm_head_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
            "lm_head",
            hidden_size,
            vocab_size,
            false,
            dtype,
            device);
    }
}

void DeepseekV4ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.kv_cache_vec.clear();
    forward_context.deepseek_v4_kv_cache_vec.clear();

    if (cache_config == nullptr) {
        cache_config_.reset();
        return;
    }

    cache_config_ = cache_config->unique_copy();
    const auto attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    auto cache_tensors = deepseek_v4_allocate_kv_cache_tensors(cache_config, model_config_, attention_backend);
    forward_context.kv_cache_vec = std::move(cache_tensors.kv_cache_tensors);
    forward_context.deepseek_v4_kv_cache_vec = std::move(cache_tensors.deepseek_v4_kv_cache_tensors);
}

infinilm::InfinilmModel::Output DeepseekV4ForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    const size_t token_count = input.input_ids.has_value() ? input.input_ids.value()->numel() : 0;
    profile::ScopedTimer forward_timer(profile::Event::CausalForward, token_count);

    infinicore::Tensor hidden_states;
    {
        profile::ScopedTimer timer(profile::Event::CausalModel, token_count);
        hidden_states = model_->forward(input);
    }

    infinicore::Tensor logits;
    {
        profile::ScopedTimer timer(profile::Event::CausalLmHead, token_count);

        const int repeats = 1; // for test 5000   // false是3868.316 true是1150(新增reduce后是1111.667)
        for (int i = 0; i < repeats; ++i) {
            logits = _compute_lm_head_logits(hidden_states);
        }
    }

    {
        profile::ScopedTimer timer(profile::Event::CausalLogitsView, token_count);
        if (logits->ndim() == 2) {
            logits = logits->view({1, logits->size(0), logits->size(1)});
        }
    }
    return {logits, hidden_states};
}

infinicore::Tensor DeepseekV4ForCausalLM::_compute_lm_head_logits(const infinicore::Tensor &hidden_states) const {
    auto mutable_hidden = hidden_states;
    if (use_parallellm_head_) {
        if (!parallel_lm_head_) {
            throw std::runtime_error("DeepseekV4ForCausalLM: parallel lm_head is not initialized.");
        }
        return parallel_lm_head_->forward(mutable_hidden);
    }
    if (!replicated_lm_head_) {
        throw std::runtime_error("DeepseekV4ForCausalLM: replicated lm_head is not initialized.");
    }
    return replicated_lm_head_->forward(mutable_hidden);
}

infinicore::Tensor DeepseekV4ForCausalLM::logits_from_hidden(const infinicore::Tensor &hidden_states) const {
    auto logits = _compute_lm_head_logits(hidden_states);
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
