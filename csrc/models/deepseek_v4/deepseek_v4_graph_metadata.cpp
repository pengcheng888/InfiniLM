#include "deepseek_v4_graph_metadata.hpp"

#include "../../global_state/global_state.hpp"

#include <cstdint>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

constexpr size_t kDsv4SwaTopk = 128;
constexpr size_t kDsv4C128MetadataWidth = 8256;

infinicore::Tensor make_i32_tensor(const std::vector<size_t> &shape,
                                   int32_t fill,
                                   infinicore::Device device) {
    auto tensor = infinicore::Tensor::empty(shape, infinicore::DataType::I32, device);
    std::vector<int32_t> values(tensor->numel(), fill);
    infinicore::context::memcpyH2D(tensor->data(), values.data(), values.size() * sizeof(int32_t), false);
    return tensor;
}

} // namespace

void init_graph_decode_metadata(infinilm::InfinilmModel::Input &input,
                                size_t batch_size,
                                size_t block_per_req,
                                infinicore::Device device) {
    auto &deepseek_v4 = input.deepseek_v4;
    deepseek_v4.swa_indices = make_i32_tensor({batch_size, kDsv4SwaTopk}, -1, device);
    deepseek_v4.swa_topk_lengths = make_i32_tensor({batch_size}, 1, device);

    deepseek_v4.raw_out_loc = make_i32_tensor({batch_size}, 0, device);
    deepseek_v4.page_table = make_i32_tensor({batch_size, block_per_req}, -1, device);
    std::vector<int32_t> page_table(batch_size * block_per_req, -1);
    for (size_t row = 0; row < batch_size; ++row) {
        page_table[row * block_per_req] = 0;
    }
    infinicore::context::memcpyH2D(deepseek_v4.page_table->data(), page_table.data(), page_table.size() * sizeof(int32_t), false);

    deepseek_v4.c4_out_loc = make_i32_tensor({batch_size}, -1, device);
    deepseek_v4.c4_positions = make_i32_tensor({batch_size}, 0, device);
    deepseek_v4.c4_topk_lengths_raw = make_i32_tensor({batch_size}, 0, device);
    deepseek_v4.c4_sparse_topk_lengths = make_i32_tensor({batch_size}, 1, device);

    deepseek_v4.c128_out_loc = make_i32_tensor({batch_size}, -1, device);
    deepseek_v4.c128_positions = make_i32_tensor({batch_size}, 0, device);
    deepseek_v4.c128_page_indices = make_i32_tensor({batch_size, kDsv4C128MetadataWidth}, -1, device);
    deepseek_v4.c128_topk_lengths_clamp1 = make_i32_tensor({batch_size}, 1, device);

    deepseek_v4.c4_compress_write_loc = make_i32_tensor({batch_size}, 0, device);
    deepseek_v4.c4_compress_extra_loc = make_i32_tensor({batch_size, 1}, 0, device);
    deepseek_v4.c128_compress_write_loc = make_i32_tensor({batch_size}, 0, device);
}

void bind_graph_forward_context_from_input(const infinilm::InfinilmModel::Input &input) {
    bind_graph_forward_context_from_input(input, infinilm::global_state::DeepSeekV4FlashMLAScheduleCache{});
}

void bind_graph_forward_context_from_input(
    const infinilm::InfinilmModel::Input &input,
    const infinilm::global_state::DeepSeekV4FlashMLAScheduleCache &schedule_cache) {
    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.attn_metadata = infinilm::global_state::AttentionMetadata(input);
    forward_context.deepseek_v4_attention_metadata = infinilm::global_state::DeepSeekV4AttentionMetadata(input);
    forward_context.deepseek_v4_flashmla_schedule_cache = schedule_cache;
}

} // namespace infinilm::models::deepseek_v4
