#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../../models/deepseek_v4/deepseek_v4_graph_metadata.hpp"
#include "../../utils.hpp"
#include "infinicore/ops/deepseek_v4_flashmla_compute.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::engine {

namespace {

constexpr int kDeepSeekV4C4SparseTopk = 512;

void bind_forward_context_from_input(const InfinilmModel::Input &input) {
    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.attn_metadata = infinilm::global_state::AttentionMetadata(input);
}

bool copy_graph_input_tensor(infinicore::Tensor &dst, const infinicore::Tensor &src) {
    if (!dst || !src) {
        return false;
    }
    if (dst->shape() == src->shape()) {
        dst->copy_from(src);
        return true;
    }
    if (dst->ndim() == 2 && src->ndim() == 2 && dst->size(0) == src->size(0) && src->size(1) <= dst->size(1)) {
        set_minus_one_device_async(dst);
        dst->narrow({{1, 0, src->size(1)}})->copy_from(src);
        return true;
    }
    return false;
}

bool copy_graph_input_optional(std::optional<infinicore::Tensor> &dst,
                               const std::optional<infinicore::Tensor> &src) {
    if (!dst.has_value() && !src.has_value()) {
        return true;
    }
    return dst.has_value() && src.has_value() && copy_graph_input_tensor(dst.value(), src.value());
}

bool refresh_flashmla_schedule(infinilm::global_state::FlashMLASchedMeta &flashmla_metadata,
                               const infinicore::Tensor &topk_lengths,
                               const infinicore::Tensor &indices,
                               std::optional<infinicore::Tensor> extra_topk_lengths = std::nullopt,
                               int extra_topk = -1) {
    auto &tile_scheduler_metadata = flashmla_metadata.tile_scheduler_metadata;
    auto &num_splits = flashmla_metadata.num_splits;
    if (!tile_scheduler_metadata || !num_splits || !topk_lengths || !indices || indices->ndim() < 2) {
        return false;
    }
    const int topk = static_cast<int>(indices->size(indices->ndim() - 1));
    if (extra_topk_lengths.has_value() && extra_topk_lengths.value()) {
        if (extra_topk <= 0) {
            return false;
        }
    } else {
        extra_topk = -1;
    }
    infinicore::op::deepseek_v4_flashmla_sparse_attention_metadata_(tile_scheduler_metadata,
                                                                    num_splits,
                                                                    topk_lengths,
                                                                    topk,
                                                                    extra_topk_lengths,
                                                                    extra_topk);
    flashmla_metadata.have_initialized = true;
    return true;
}

bool refresh_deepseek_v4_flashmla_schedules(
    InfinilmModel::Input &graph_input,
    infinilm::global_state::DSV4AttnMetadata &dsv4_metadata) {
    return refresh_flashmla_schedule(dsv4_metadata.c1_flashmla_metadata,
                                     graph_input.deepseek_v4.swa_topk_lengths,
                                     graph_input.deepseek_v4.swa_indices)
        && refresh_flashmla_schedule(dsv4_metadata.c4_flashmla_metadata,
                                     graph_input.deepseek_v4.swa_topk_lengths,
                                     graph_input.deepseek_v4.swa_indices,
                                     graph_input.deepseek_v4.c4_sparse_topk_lengths,
                                     kDeepSeekV4C4SparseTopk)
        && refresh_flashmla_schedule(dsv4_metadata.c128_flashmla_metadata,
                                     graph_input.deepseek_v4.swa_topk_lengths,
                                     graph_input.deepseek_v4.swa_indices,
                                     graph_input.deepseek_v4.c128_topk_lengths_clamp1,
                                     static_cast<int>(graph_input.deepseek_v4.c128_page_indices->size(
                                         graph_input.deepseek_v4.c128_page_indices->ndim() - 1)));
}

} // namespace

PagedCompiler::PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
    const bool is_deepseek_v4 = model_ && model_->model_type() == "deepseek_v4";
    if (is_deepseek_v4) {
        for (size_t b = 1; b <= 32; ++b) {
            decode_batch_sizes_.push_back(b);
        }
        return;
    }

    for (size_t b = 1; b < 64; ++b) {
        decode_batch_sizes_.push_back(b);
    }
    for (size_t b = 64; b < 128; b += 16) {
        decode_batch_sizes_.push_back(b);
    }
    for (size_t b = 128; b < 256; b += 32) {
        decode_batch_sizes_.push_back(b);
    }
    for (size_t b = 256; b <= 512; b += 64) {
        decode_batch_sizes_.push_back(b);
    }
}

void PagedCompiler::compile() {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        size_t nblocks = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
        size_t max_batch_size = *std::max_element(decode_batch_sizes_.begin(), decode_batch_sizes_.end());
        const bool is_deepseek_v4_model = model_ && model_->model_type() == "deepseek_v4";
        compiled_map_decode_.clear();
        block_tables_holder_ = infinicore::Tensor::empty(
            {nblocks * max_batch_size}, infinicore::DataType::I32, infinicore::context::getDevice());
        set_zeros(block_tables_holder_);

        auto make_decode_input = [&](size_t b) {
            const bool is_deepseek_v4 = model_ && model_->model_type() == "deepseek_v4";
            InfinilmModel::Input input;
            input.input_ids = infinicore::Tensor::empty({1, b}, infinicore::DataType::I64, infinicore::context::getDevice());
            input.position_ids = infinicore::Tensor::empty({b}, infinicore::DataType::I64, infinicore::context::getDevice());
            input.total_sequence_lengths = infinicore::Tensor::empty({b}, infinicore::DataType::I32, infinicore::context::getDevice());
            if (is_deepseek_v4) {
                // Token 104937 is a known-valid DeepSeek-V4 decode token from
                // the local correctness tests. Avoid token-id 0 here because
                // FFN graph capture now exercises hash-topk and MoE kernels.
                std::vector<int64_t> input_ids_vec(b, 104937);
                infinicore::context::memcpyH2D(input.input_ids.value()->data(), input_ids_vec.data(), b * sizeof(int64_t), false);
            } else {
                set_zeros(input.input_ids.value());
            }
            set_zeros(input.position_ids.value());
            set_zeros(input.total_sequence_lengths.value());
            std::vector<int32_t> total_sequence_lengths_vec(b, 1);
            infinicore::context::memcpyH2D(input.total_sequence_lengths.value()->data(), total_sequence_lengths_vec.data(), b * sizeof(int32_t), false);
            input.input_offsets = infinicore::Tensor::empty({b + 1}, infinicore::DataType::I32, infinicore::context::getDevice());
            std::vector<int32_t> input_offsets_vec(b + 1, 0);
            for (size_t i = 0; i <= b; i++) {
                input_offsets_vec[i] = i;
            }
            infinicore::context::memcpyH2D(input.input_offsets.value()->data(), input_offsets_vec.data(), (b + 1) * sizeof(int32_t), false);
            input.cu_seqlens = infinicore::Tensor::empty({b + 1}, infinicore::DataType::I32, infinicore::context::getDevice());
            infinicore::context::memcpyH2D(input.cu_seqlens.value()->data(), input_offsets_vec.data(), (b + 1) * sizeof(int32_t), false);
            const size_t block_per_req = nblocks;
            input.block_tables = block_tables_holder_->as_strided({b, block_per_req}, {(ptrdiff_t)block_per_req, 1});
            input.slot_mapping = infinicore::Tensor::empty({b}, infinicore::DataType::I64, infinicore::context::getDevice());
            set_zeros(input.slot_mapping.value());

            if (is_deepseek_v4) {
                infinilm::models::deepseek_v4::init_graph_decode_metadata(input, b, block_per_req, infinicore::context::getDevice());
            }

            // Attention reads metadata from thread-local forward context.
            if (is_deepseek_v4) {
                infinilm::models::deepseek_v4::bind_graph_forward_context_from_input(input);
            } else {
                bind_forward_context_from_input(input);
            }
            return input;
        };

        {
            const size_t warmup_batch_size = std::min(max_batch_size, static_cast<size_t>(64));
            auto input = make_decode_input(warmup_batch_size);
            model_->forward(input);
            infinicore::context::syncStream();
            // Warmup runs the eager Marlin path and may leave per-layer lock
            // workspaces dirty. Reset before CUDA graph capture so capture
            // starts from the same all-zero lock state as normal execution.
            model_->reset_runtime_state();
            infinicore::context::syncStream();
        }

        for (size_t b : decode_batch_sizes_) {
            auto input = make_decode_input(b);

            barrier_->wait();
            (void)model_->forward(input);
            infinicore::context::syncStream();
            infinilm::global_state::DSV4AttnMetadata dsv4_attn_metadata;
            if (is_deepseek_v4_model) {
                dsv4_attn_metadata = infinilm::global_state::get_forward_context().dsv4_attn_metadata;
                infinilm::models::deepseek_v4::bind_graph_forward_context_from_input(input, dsv4_attn_metadata);
            }
            // Capture must not start with stale Marlin locks from previous
            // warmup/capture attempts. This reset is intentionally outside
            // graph capture; the current implementation still pays a memset
            // before every graph replay in get_compiled().
            model_->reset_runtime_state();
            infinicore::context::syncStream();
            infinicore::context::startGraphRecording();
            if (is_deepseek_v4_model &&
                !refresh_deepseek_v4_flashmla_schedules(input, dsv4_attn_metadata)) {
                infinicore::context::stopGraphRecording();
                throw std::runtime_error("failed to record DeepSeek-V4 FlashMLA schedule metadata refresh");
            }
            auto output = model_->forward(input);
            auto graph = infinicore::context::stopGraphRecording();
            barrier_->wait();

            auto shared_output = std::shared_ptr<InfinilmModel::Output>(
                new InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});

            compiled_map_decode_[b] = CompiledResult{
                std::move(input),
                std::move(dsv4_attn_metadata),
                std::make_tuple(graph, shared_output)};
        }
    }
}

PagedCompiler::Compiled PagedCompiler::get_compiled(const InfinilmModel::Input &input) {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        size_t batch_size = input.block_tables.value()->size(0);
        size_t block_per_req = input.block_tables.value()->size(1);

        // only support decode only batch
        if (batch_size != input.input_ids.value()->size(1)) {
            return {nullptr, nullptr};
        } else {
            auto result = compiled_map_decode_.find(batch_size);
            if (result == compiled_map_decode_.end()) {
                return {nullptr, nullptr};
            }
            auto &graph_input = result->second.input;

            graph_input.input_ids.value()->copy_from(input.input_ids.value());
            graph_input.position_ids.value()->copy_from(input.position_ids.value());
            graph_input.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());
            graph_input.input_offsets.value()->copy_from(input.input_offsets.value());
            graph_input.cu_seqlens.value()->copy_from(input.cu_seqlens.value());

            const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
            if (block_per_req > compiled_block_per_req) {
                // Runtime width exceeds compiled graph slot; fall back to eager path.
                return {nullptr, nullptr};
            }

            // Initialize only the active graph rows to -1, then overwrite the
            // runtime logical region. Avoid clearing the full preallocated
            // holder on every decode token.
            auto &graph_block_tables = graph_input.block_tables.value();
            set_minus_one_device_async(graph_block_tables);
            graph_block_tables->narrow({{1, 0, block_per_req}})->copy_from(input.block_tables.value());
            graph_input.slot_mapping.value()->copy_from(input.slot_mapping.value());

            if (model_ && model_->model_type() == "deepseek_v4") {
                const bool dsv4_copied = copy_graph_input_tensor(graph_input.deepseek_v4.swa_indices, input.deepseek_v4.swa_indices)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.swa_topk_lengths, input.deepseek_v4.swa_topk_lengths)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.raw_out_loc, input.deepseek_v4.raw_out_loc)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.page_table, input.deepseek_v4.page_table)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.c4_out_loc, input.deepseek_v4.c4_out_loc)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.c4_positions, input.deepseek_v4.c4_positions)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.c4_topk_lengths_raw, input.deepseek_v4.c4_topk_lengths_raw)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.c4_sparse_topk_lengths, input.deepseek_v4.c4_sparse_topk_lengths)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.c128_out_loc, input.deepseek_v4.c128_out_loc)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.c128_positions, input.deepseek_v4.c128_positions)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.c128_page_indices, input.deepseek_v4.c128_page_indices)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.c128_topk_lengths_clamp1, input.deepseek_v4.c128_topk_lengths_clamp1)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.c4_compress_write_loc, input.deepseek_v4.c4_compress_write_loc)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.c4_compress_extra_loc, input.deepseek_v4.c4_compress_extra_loc)
                                      && copy_graph_input_tensor(graph_input.deepseek_v4.c128_compress_write_loc, input.deepseek_v4.c128_compress_write_loc);
                if (!dsv4_copied) {
                    return {nullptr, nullptr};
                }
            }
            // CUDA graph replay reuses the same per-layer Marlin workspaces.
            // The graph itself does not contain a workspace reset, so enqueue
            // one on the same stream before launch. This is correct but costs
            // decode latency; the intended follow-up is a reusable global
            // zero workspace/lock buffer shared by all Marlin layers.
            model_->reset_runtime_state();

            auto graph = std::get<0>(result->second.compiled);
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{std::get<1>(result->second.compiled)->logits->resume_from_blob_()});

            return std::make_tuple(graph, shared_output);
        }
    } else {
        return {nullptr, nullptr};
    }
}

} // namespace infinilm::engine
