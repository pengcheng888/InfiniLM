#include "deepseek_v4_mlp.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops/swiglu.hpp"

namespace infinilm::models::deepseek_v4 {

DeepseekV4MLP::DeepseekV4MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device) {
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("intermediate_size");
    const auto quantization_method = model_config->get_quantization_method();
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    auto register_fn = [this](const std::string &name, infinicore::nn::Parameter param) {
        this->register_parameter(name, std::move(param));
    };

    gate_up_proj_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
        hidden_size,
        intermediate_size,
        "w1",
        "w3",
        register_fn,
        quantization_method,
        false,
        dtype,
        device,
        rank_info);
    w2_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "w2",
        intermediate_size,
        hidden_size,
        quantization_method,
        false,
        dtype,
        device,
        static_cast<infinicore::Size>(rank_info.tp_rank),
        static_cast<infinicore::Size>(rank_info.tp_size),
        nullptr);
}

infinicore::Tensor DeepseekV4MLP::forward(const infinicore::Tensor &hidden_states) const {
    auto input = hidden_states;
    auto [gate, up] = gate_up_proj_->forward_split(input);
    auto activated = infinicore::op::swiglu(up, gate);
    return w2_->forward(activated);
}

void DeepseekV4MLP::process_weights_after_loading() {
    gate_up_proj_->process_weights_after_loading();
    w2_->process_weights_after_loading();
}

void DeepseekV4MLP::reset_runtime_state() const {
    gate_up_proj_->reset_runtime_state();
    w2_->reset_runtime_state();
}

} // namespace infinilm::models::deepseek_v4
