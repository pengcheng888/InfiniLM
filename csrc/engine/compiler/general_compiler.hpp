#pragma once

#include "paged_compiler.hpp"
#include "static_batching_compiler.hpp"

namespace infinilm::engine {
class GeneralCompiler : public GraphCompiler {
public:
    GeneralCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile() override;
    // 同时释放 static 与 paged 两条编译路径缓存的 graph。
    void clear() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

private:
    std::unique_ptr<StaticBatchingCompiler> static_batching_compiler_;
    std::unique_ptr<PagedCompiler> paged_compiler_;
};
} // namespace infinilm::engine
