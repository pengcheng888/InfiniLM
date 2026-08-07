#include "qwen3_mlp.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops/qwen3_silu_and_mul.hpp"

namespace infinilm::models::qwen3 {

Qwen3MLP::Qwen3MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    hidden_size_ = model_config->get<size_t>("hidden_size");
    intermediate_size_ = model_config->get<size_t>("intermediate_size");
    use_bias_ = model_config->get_or<bool>("mlp_bias", false);

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) {
        this->register_parameter(n, std::move(p));
    };
    gate_up_proj_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
        hidden_size_, intermediate_size_, "gate_proj", "up_proj", register_fn,
        quantization_method, use_bias_, dtype, device, rank_info);
    down_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "down_proj", intermediate_size_, hidden_size_, quantization_method,
        use_bias_, dtype, device, tp_rank, tp_size, rank_info.comm);
}

infinicore::Tensor Qwen3MLP::forward(const infinicore::Tensor &hidden_states) const {
    auto input = hidden_states;
    auto gate_up = gate_up_proj_->forward(input);
    auto intermediate = infinicore::op::qwen3_silu_and_mul(gate_up);
    return down_proj_->forward(intermediate);
}

} // namespace infinilm::models::qwen3
