#pragma once

#include "../../global_state/forward_context.hpp"
#include "../infinilm_model.hpp"

#include <cstddef>

namespace infinilm::models::deepseek_v4 {

void init_graph_decode_metadata(infinilm::InfinilmModel::Input &input,
                                size_t batch_size,
                                size_t block_per_req,
                                infinicore::Device device);

void bind_graph_forward_context_from_input(const infinilm::InfinilmModel::Input &input);

void bind_graph_forward_context_from_input(
    const infinilm::InfinilmModel::Input &input,
    const infinilm::global_state::DeepSeekV4FlashMLAScheduleCache &schedule_cache);

} // namespace infinilm::models::deepseek_v4
