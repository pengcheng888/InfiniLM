#include "rotary_embedding.hpp"
#include <string>
#include <unordered_map>

namespace infinilm::layers::rotary_embedding {
namespace {
thread_local std::unordered_map<std::string, std::shared_ptr<infinicore::nn::RoPE>> _ROPE_DICT;

} // namespace

std::shared_ptr<infinicore::nn::RoPE> get_rope(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                                               const infinicore::Device &device) {
    const std::string scaling_type = "default";
    auto it = _ROPE_DICT.find(scaling_type);
    if (it != _ROPE_DICT.end()) {
        return it->second;
    }

    const auto &dtype = model_config->get_dtype();
    size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    double rope_theta = model_config->get<double>("rope_theta");
    auto rope = std::make_shared<infinicore::nn::RoPE>(model_config->get_head_dim(), max_position_embeddings, rope_theta,
                                                       infinicore::nn::RoPE::Algo::GPT_NEOX, dtype, device,
                                                       model_config->get_rope_scaling());

    _ROPE_DICT.emplace(scaling_type, rope);
    return rope;
}

} // namespace infinilm::layers::rotary_embedding
