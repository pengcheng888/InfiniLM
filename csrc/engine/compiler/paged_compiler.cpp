#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <cstdint>
#include <vector>

namespace infinilm::engine {

namespace {

constexpr size_t kDsv4SwaTopk = 128;
constexpr size_t kDsv4C4Topk = 512;
constexpr size_t kDsv4C128MetadataWidth = 8256;

infinicore::Tensor make_i32_tensor(const std::vector<size_t> &shape,
                                   int32_t fill,
                                   infinicore::Device device) {
    auto tensor = infinicore::Tensor::empty(shape, infinicore::DataType::I32, device);
    std::vector<int32_t> values(tensor->numel(), fill);
    infinicore::context::memcpyH2D(tensor->data(), values.data(), values.size() * sizeof(int32_t), false);
    return tensor;
}

void init_deepseek_v4_graph_decode_metadata(InfinilmModel::Input &input,
                                            size_t batch_size,
                                            size_t block_per_req,
                                            infinicore::Device device) {
    input.dsv4_swa_indices = make_i32_tensor({batch_size, kDsv4SwaTopk}, -1, device);
    input.dsv4_swa_topk_lengths = make_i32_tensor({batch_size}, 1, device);
    input.dsv4_c4_indices = make_i32_tensor({batch_size, kDsv4C4Topk}, -1, device);
    input.dsv4_c4_topk_lengths = make_i32_tensor({batch_size}, 1, device);
    input.dsv4_c128_indices = make_i32_tensor({batch_size, kDsv4C128MetadataWidth}, -1, device);
    input.dsv4_c128_topk_lengths = make_i32_tensor({batch_size}, 1, device);

    input.dsv4_raw_out_loc = make_i32_tensor({batch_size}, 0, device);
    input.dsv4_page_table = make_i32_tensor({batch_size, block_per_req}, -1, device);
    std::vector<int32_t> page_table(batch_size * block_per_req, -1);
    for (size_t row = 0; row < batch_size; ++row) {
        page_table[row * block_per_req] = 0;
    }
    infinicore::context::memcpyH2D(input.dsv4_page_table.value()->data(), page_table.data(), page_table.size() * sizeof(int32_t), false);
    input.dsv4_seq_lens_casual = make_i32_tensor({batch_size}, 1, device);
    input.dsv4_positions_casual = make_i32_tensor({batch_size}, 0, device);

    input.dsv4_c4_out_loc = make_i32_tensor({batch_size}, -1, device);
    input.dsv4_c4_positions = make_i32_tensor({batch_size}, 0, device);
    input.dsv4_c4_topk_lengths_raw = make_i32_tensor({batch_size}, 0, device);
    input.dsv4_c4_topk_lengths_clamp1 = make_i32_tensor({batch_size}, 1, device);
    input.dsv4_c4_sparse_indices = make_i32_tensor({batch_size, kDsv4C4Topk}, -1, device);
    input.dsv4_c4_sparse_topk_lengths = make_i32_tensor({batch_size}, 1, device);

    input.dsv4_c128_out_loc = make_i32_tensor({batch_size}, -1, device);
    input.dsv4_c128_positions = make_i32_tensor({batch_size}, 0, device);
    input.dsv4_c128_page_indices = make_i32_tensor({batch_size, kDsv4C128MetadataWidth}, -1, device);
    input.dsv4_c128_topk_lengths_clamp1 = make_i32_tensor({batch_size}, 1, device);

    input.dsv4_c4_compress_write_loc = make_i32_tensor({batch_size}, 0, device);
    input.dsv4_c4_compress_extra_loc = make_i32_tensor({batch_size, 1}, 0, device);
    input.dsv4_c4_compress_state_indices = make_i32_tensor({batch_size}, 0, device);
    input.dsv4_c128_compress_write_loc = make_i32_tensor({batch_size}, 0, device);
    input.dsv4_c128_compress_state_indices = make_i32_tensor({batch_size}, 0, device);
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

} // namespace

PagedCompiler::PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
    const bool is_deepseek_v4 = model_ && model_->model_type() == "deepseek_v4";
    if (is_deepseek_v4) {
        for (size_t b = 32; b >= 1; --b) {
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
                init_deepseek_v4_graph_decode_metadata(input, b, block_per_req, infinicore::context::getDevice());
            }

            // Attention reads attn_metadata from thread-local forward context.
            infinilm::global_state::get_forward_context().attn_metadata = infinilm::global_state::AttentionMetadata(input);
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
            // Capture must not start with stale Marlin locks from previous
            // warmup/capture attempts. This reset is intentionally outside
            // graph capture; the current implementation still pays a memset
            // before every graph replay in get_compiled().
            model_->reset_runtime_state();
            infinicore::context::syncStream();
            infinicore::context::startGraphRecording();
            auto output = model_->forward(input);
            auto graph = infinicore::context::stopGraphRecording();
            barrier_->wait();

            auto shared_output = std::shared_ptr<InfinilmModel::Output>(
                new InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});

            compiled_map_decode_[b] = CompiledResult{std::move(input), std::make_tuple(graph, shared_output)};
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

            const bool dsv4_copied = copy_graph_input_optional(graph_input.dsv4_swa_indices, input.dsv4_swa_indices)
                                  && copy_graph_input_optional(graph_input.dsv4_swa_topk_lengths, input.dsv4_swa_topk_lengths)
                                  && copy_graph_input_optional(graph_input.dsv4_c4_indices, input.dsv4_c4_indices)
                                  && copy_graph_input_optional(graph_input.dsv4_c4_topk_lengths, input.dsv4_c4_topk_lengths) && copy_graph_input_optional(graph_input.dsv4_c128_indices, input.dsv4_c128_indices) && copy_graph_input_optional(graph_input.dsv4_c128_topk_lengths, input.dsv4_c128_topk_lengths)
                                  && copy_graph_input_optional(graph_input.dsv4_raw_out_loc, input.dsv4_raw_out_loc)
                                  && copy_graph_input_optional(graph_input.dsv4_page_table, input.dsv4_page_table)
                                  && copy_graph_input_optional(graph_input.dsv4_seq_lens_casual, input.dsv4_seq_lens_casual)
                                  && copy_graph_input_optional(graph_input.dsv4_positions_casual, input.dsv4_positions_casual)
                                  && copy_graph_input_optional(graph_input.dsv4_c4_out_loc, input.dsv4_c4_out_loc)
                                  && copy_graph_input_optional(graph_input.dsv4_c4_positions, input.dsv4_c4_positions)
                                  && copy_graph_input_optional(graph_input.dsv4_c4_topk_lengths_raw, input.dsv4_c4_topk_lengths_raw)
                                  && copy_graph_input_optional(graph_input.dsv4_c4_topk_lengths_clamp1, input.dsv4_c4_topk_lengths_clamp1)
                                  && copy_graph_input_optional(graph_input.dsv4_c4_sparse_indices, input.dsv4_c4_sparse_indices)
                                  && copy_graph_input_optional(graph_input.dsv4_c4_sparse_topk_lengths, input.dsv4_c4_sparse_topk_lengths)
                                  && copy_graph_input_optional(graph_input.dsv4_c128_out_loc, input.dsv4_c128_out_loc)
                                  && copy_graph_input_optional(graph_input.dsv4_c128_positions, input.dsv4_c128_positions)
                                  && copy_graph_input_optional(graph_input.dsv4_c128_page_indices, input.dsv4_c128_page_indices)
                                  && copy_graph_input_optional(graph_input.dsv4_c128_topk_lengths_clamp1, input.dsv4_c128_topk_lengths_clamp1)
                                  && copy_graph_input_optional(graph_input.dsv4_c4_compress_write_loc, input.dsv4_c4_compress_write_loc)
                                  && copy_graph_input_optional(graph_input.dsv4_c4_compress_extra_loc, input.dsv4_c4_compress_extra_loc)
                                  && copy_graph_input_optional(graph_input.dsv4_c4_compress_state_indices, input.dsv4_c4_compress_state_indices)
                                  && copy_graph_input_optional(graph_input.dsv4_c128_compress_write_loc, input.dsv4_c128_compress_write_loc)
                                  && copy_graph_input_optional(graph_input.dsv4_c128_compress_state_indices, input.dsv4_c128_compress_state_indices);
            if (!dsv4_copied) {
                return {nullptr, nullptr};
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
