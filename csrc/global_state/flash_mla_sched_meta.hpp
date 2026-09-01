#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>
#include <optional>

namespace infinilm::global_state {

struct FlashMLASchedMeta {
    struct Config {
        size_t b;
        size_t s_q;
        size_t h_q;
        size_t page_block_size;
        size_t h_k;

        bool causal;
        bool is_fp8_kvcache;
        std::optional<size_t> topk;

        std::optional<size_t> extra_page_block_size;
        std::optional<size_t> extra_topk;
    };

    bool have_initialized{false};
    bool have_refreshed{false};
    std::optional<Config> config;
    infinicore::Tensor tile_scheduler_metadata;
    infinicore::Tensor num_splits;
    //
    infinicore::Tensor out;
    infinicore::Tensor lse;
    infinicore::Tensor lse_accum;
    infinicore::Tensor o_accum;

    FlashMLASchedMeta() = default;

    bool has_sched_buffer() const {
        return tile_scheduler_metadata && num_splits;
    }

    bool has_valid_sched_meta() const {
        return has_sched_buffer() && have_refreshed;
    }

    bool has_sched_meta() const {
        return has_sched_buffer();
    }
};

} // namespace infinilm::global_state
