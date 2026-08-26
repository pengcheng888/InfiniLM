#pragma once

#include "../../global_state/forward_context.hpp"
#include "graph_compiler.hpp"

#include <unordered_map>
#include <vector>

namespace infinilm::engine {
class PagedCompiler : public GraphCompiler {
public:
    PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

private:
    std::vector<size_t> decode_batch_sizes_;

    infinicore::Tensor block_tables_holder_;

    struct CompiledResult {
        InfinilmModel::Input input;
        std::vector<infinilm::global_state::FlashMLASchedMeta> flashmla_sched_meta_vec;
        infinilm::global_state::DSV4AttnMetadata dsv4_attn_metadata;
        Compiled compiled;
    };

    std::unordered_map<
        size_t, // num_requests
        CompiledResult>
        compiled_map_decode_;
};
} // namespace infinilm::engine
