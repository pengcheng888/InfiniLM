#pragma once

#include "../models/infinilm_model.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace infinilm::engine {

infinilm::DeepSeekV4Input build_deepseek_v4_attention_metadata(
    const infinicore::Tensor &block_tables,
    const infinicore::Tensor &slot_mapping,
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &input_offsets,
    std::optional<std::vector<int64_t>> full_to_swa_block_ids,
    size_t block_size);

} // namespace infinilm::engine
