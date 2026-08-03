#include "deepseek_v4_metadata.hpp"

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace infinilm::engine {
namespace {

infinicore::Tensor cpu_contiguous(const infinicore::Tensor &tensor, const char *name) {
    if (!tensor) {
        throw std::runtime_error(std::string("build_deepseek_v4_attention_metadata: missing ") + name);
    }
    auto result = tensor;
    if (result->device().getType() != infinicore::Device::Type::CPU) {
        result = result->to(infinicore::Device::cpu());
    }
    if (!result->is_contiguous()) {
        result = result->contiguous();
    }
    return result;
}

void require_rank(const infinicore::Tensor &tensor, size_t rank, const char *name) {
    if (tensor->ndim() != rank) {
        throw std::runtime_error(std::string("build_deepseek_v4_attention_metadata: ") + name
                                 + " rank mismatch");
    }
}

void require_dtype(const infinicore::Tensor &tensor, infinicore::DataType dtype, const char *name) {
    if (tensor->dtype() != dtype) {
        throw std::runtime_error(std::string("build_deepseek_v4_attention_metadata: ") + name
                                 + " dtype mismatch");
    }
}

int64_t read_position(const std::byte *data, infinicore::DataType dtype, size_t idx) {
    if (dtype == infinicore::DataType::I64) {
        return reinterpret_cast<const int64_t *>(data)[idx];
    }
    if (dtype == infinicore::DataType::I32) {
        return reinterpret_cast<const int32_t *>(data)[idx];
    }
    throw std::runtime_error("build_deepseek_v4_attention_metadata: position_ids must be int32 or int64");
}

size_t align_up(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

int64_t slot_for_position(const int32_t *block_table,
                          size_t block_table_len,
                          int64_t position,
                          size_t block_size) {
    if (position < 0) {
        return -1;
    }
    const auto block_idx = static_cast<size_t>(position) / block_size;
    const auto block_offset = static_cast<size_t>(position) % block_size;
    if (block_idx >= block_table_len) {
        return -1;
    }
    const int32_t block_id = block_table[block_idx];
    return block_id >= 0 ? static_cast<int64_t>(block_id) * static_cast<int64_t>(block_size)
                               + static_cast<int64_t>(block_offset)
                         : -1;
}

int64_t full_slot_to_swa_slot(int64_t slot,
                              const std::optional<std::vector<int64_t>> &full_to_swa_block_ids,
                              size_t block_size) {
    if (slot < 0) {
        return -1;
    }
    if (!full_to_swa_block_ids.has_value()) {
        return slot;
    }
    const auto full_block_id = static_cast<size_t>(slot) / block_size;
    const auto block_offset = static_cast<size_t>(slot) % block_size;
    if (full_block_id >= full_to_swa_block_ids->size()) {
        return -1;
    }
    const int64_t swa_block_id = full_to_swa_block_ids->at(full_block_id);
    return swa_block_id >= 0 ? swa_block_id * static_cast<int64_t>(block_size)
                                   + static_cast<int64_t>(block_offset)
                             : -1;
}

int64_t state_loc(int64_t raw_slot, size_t ratio, size_t block_size) {
    if (raw_slot < 0) {
        return -1;
    }
    const size_t ring_size = ratio == 4 ? 8 : 128;
    const auto slot = static_cast<size_t>(raw_slot);
    return static_cast<int64_t>(((slot / block_size) * ring_size + (slot % ring_size)) / ratio);
}

struct CompressLocations {
    int64_t out_loc;
    int64_t clipped_position;
    int64_t write_loc;
    int64_t prev_state_index;
};

CompressLocations compress_locations_for_position(
    const int32_t *block_table,
    size_t block_table_len,
    int64_t slot,
    int64_t position,
    size_t ratio,
    size_t block_size,
    const std::optional<std::vector<int64_t>> &full_to_state_block_ids) {
    const int64_t state_slot = full_slot_to_swa_slot(slot, full_to_state_block_ids, block_size);
    const int64_t state_index = state_loc(state_slot, ratio, block_size);
    const int64_t out_loc = slot >= 0 && (position + 1) % static_cast<int64_t>(ratio) == 0
                                ? slot / static_cast<int64_t>(ratio)
                                : -1;
    const int64_t clipped_position = (position / static_cast<int64_t>(ratio))
                                   * static_cast<int64_t>(ratio);
    const int64_t clipped_slot = slot_for_position(block_table, block_table_len, clipped_position, block_size);
    const int64_t clipped_state_slot = full_slot_to_swa_slot(clipped_slot, full_to_state_block_ids, block_size);
    int64_t write_loc = state_loc(clipped_state_slot, ratio, block_size);
    if (write_loc < 0) {
        write_loc = state_index;
    }

    const int64_t prev_position = clipped_position - static_cast<int64_t>(ratio);
    const int64_t prev_slot = slot_for_position(block_table, block_table_len, prev_position, block_size);
    const int64_t prev_state_slot = full_slot_to_swa_slot(prev_slot, full_to_state_block_ids, block_size);
    int64_t prev_state_index = state_loc(prev_state_slot, ratio, block_size);
    if (prev_state_index < 0) {
        prev_state_index = write_loc >= 0 ? write_loc : 0;
    }
    return {out_loc, clipped_position, write_loc, prev_state_index};
}

infinicore::Tensor make_i32_tensor(const std::vector<size_t> &shape) {
    return infinicore::Tensor::empty(shape, infinicore::DataType::I32, infinicore::Device::cpu());
}

int32_t *i32_data(infinicore::Tensor &tensor) {
    return reinterpret_cast<int32_t *>(tensor->data());
}

int32_t checked_i32(int64_t value, const char *name) {
    if (value < static_cast<int64_t>(std::numeric_limits<int32_t>::min())
        || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(std::string("build_deepseek_v4_attention_metadata: ") + name
                                 + " overflows int32");
    }
    return static_cast<int32_t>(value);
}

std::vector<int32_t> build_mapped_slots_by_position(
    const int32_t *block_table,
    size_t block_table_len,
    size_t max_position,
    size_t block_size,
    const std::optional<std::vector<int64_t>> &full_to_swa_block_ids) {
    std::vector<int32_t> slots(max_position + 1);
    for (size_t pos = 0; pos <= max_position; ++pos) {
        const int64_t full_slot = slot_for_position(
            block_table, block_table_len, static_cast<int64_t>(pos), block_size);
        slots[pos] = checked_i32(full_slot_to_swa_slot(full_slot, full_to_swa_block_ids, block_size),
                                 "mapped_slots");
    }
    return slots;
}

int32_t mapped_slot_for_position(
    const std::vector<int32_t> &mapped_slots,
    const int32_t *block_table,
    size_t block_table_len,
    int64_t position,
    size_t block_size,
    const std::optional<std::vector<int64_t>> &full_to_swa_block_ids) {
    if (position < 0) {
        return -1;
    }
    if (static_cast<size_t>(position) < mapped_slots.size()) {
        return mapped_slots[static_cast<size_t>(position)];
    }
    const int64_t full_slot = slot_for_position(block_table, block_table_len, position, block_size);
    return checked_i32(full_slot_to_swa_slot(full_slot, full_to_swa_block_ids, block_size),
                       "mapped_slot_for_position");
}

std::vector<int32_t> build_state_locs_by_position(const std::vector<int32_t> &mapped_slots,
                                                  size_t ratio,
                                                  size_t block_size) {
    std::vector<int32_t> result(mapped_slots.size());
    for (size_t pos = 0; pos < mapped_slots.size(); ++pos) {
        result[pos] = checked_i32(state_loc(mapped_slots[pos], ratio, block_size), "state_locs");
    }
    return result;
}

int32_t state_loc_for_mapped_position(
    const std::vector<int32_t> &state_locs,
    const std::vector<int32_t> &mapped_slots,
    const int32_t *block_table,
    size_t block_table_len,
    int64_t position,
    size_t ratio,
    size_t block_size,
    const std::optional<std::vector<int64_t>> &full_to_swa_block_ids) {
    if (position < 0) {
        return -1;
    }
    if (static_cast<size_t>(position) < state_locs.size()) {
        return state_locs[static_cast<size_t>(position)];
    }
    const int32_t mapped_slot = mapped_slot_for_position(
        mapped_slots, block_table, block_table_len, position, block_size, full_to_swa_block_ids);
    return checked_i32(state_loc(mapped_slot, ratio, block_size), "state_loc_for_mapped_position");
}

} // namespace

infinilm::DeepSeekV4Input build_deepseek_v4_attention_metadata(
    const infinicore::Tensor &block_tables_in,
    const infinicore::Tensor &slot_mapping_in,
    const infinicore::Tensor &position_ids_in,
    const infinicore::Tensor &input_offsets_in,
    std::optional<std::vector<int64_t>> full_to_swa_block_ids,
    size_t block_size) {
    if (block_size == 0) {
        throw std::runtime_error("build_deepseek_v4_attention_metadata: block_size must be > 0");
    }

    auto block_tables = cpu_contiguous(block_tables_in, "block_tables");
    auto slot_mapping = cpu_contiguous(slot_mapping_in, "slot_mapping");
    auto position_ids = cpu_contiguous(position_ids_in, "position_ids");
    auto input_offsets = cpu_contiguous(input_offsets_in, "input_offsets");

    require_rank(block_tables, 2, "block_tables");
    require_rank(slot_mapping, 1, "slot_mapping");
    require_rank(input_offsets, 1, "input_offsets");
    require_dtype(block_tables, infinicore::DataType::I32, "block_tables");
    require_dtype(slot_mapping, infinicore::DataType::I64, "slot_mapping");
    require_dtype(input_offsets, infinicore::DataType::I32, "input_offsets");

    const size_t req_count = block_tables->size(0);
    const size_t max_block_table_len = block_tables->size(1);
    const size_t rows = slot_mapping->numel();
    if (position_ids->numel() != rows) {
        throw std::runtime_error("build_deepseek_v4_attention_metadata: position_ids and slot_mapping length mismatch");
    }
    if (input_offsets->numel() != req_count + 1) {
        throw std::runtime_error("build_deepseek_v4_attention_metadata: input_offsets length mismatch");
    }
    if (rows == 0) {
        throw std::runtime_error("build_deepseek_v4_attention_metadata: empty query rows");
    }

    const auto *block_tables_ptr = reinterpret_cast<const int32_t *>(block_tables->data());
    const auto *slot_mapping_ptr = reinterpret_cast<const int64_t *>(slot_mapping->data());
    const auto *input_offsets_ptr = reinterpret_cast<const int32_t *>(input_offsets->data());
    const auto *position_ids_ptr = position_ids->data();

    size_t max_c128_visible = 1;
    for (size_t row = 0; row < rows; ++row) {
        const int64_t position = read_position(position_ids_ptr, position_ids->dtype(), row);
        max_c128_visible = std::max(max_c128_visible, static_cast<size_t>((position + 1) / 128));
    }
    const size_t c128_width = align_up(std::max(max_c128_visible, static_cast<size_t>(1)), 64);
    constexpr size_t swa_window_size = 128;
    constexpr size_t c4_sparse_topk = 512;

    auto swa_indices = make_i32_tensor({rows, swa_window_size});
    auto swa_topk_lengths = make_i32_tensor({rows});
    auto raw_out_loc = make_i32_tensor({rows});
    auto page_table = make_i32_tensor({rows, max_block_table_len});

    auto c4_topk_lengths_raw = make_i32_tensor({rows});
    auto c4_sparse_topk_lengths = make_i32_tensor({rows});
    auto c4_out_loc = make_i32_tensor({rows});
    auto c4_positions = make_i32_tensor({rows});
    auto c4_compress_write_loc = make_i32_tensor({rows});
    auto c4_compress_extra_loc = make_i32_tensor({rows, 1});

    auto c128_page_indices = make_i32_tensor({rows, c128_width});
    auto c128_topk_lengths_clamp1 = make_i32_tensor({rows});
    auto c128_out_loc = make_i32_tensor({rows});
    auto c128_positions = make_i32_tensor({rows});
    auto c128_compress_write_loc = make_i32_tensor({rows});

    auto *swa_indices_ptr = i32_data(swa_indices);
    auto *swa_topk_lengths_ptr = i32_data(swa_topk_lengths);
    auto *raw_out_loc_ptr = i32_data(raw_out_loc);
    auto *page_table_ptr = i32_data(page_table);
    auto *c4_topk_lengths_raw_ptr = i32_data(c4_topk_lengths_raw);
    auto *c4_sparse_topk_lengths_ptr = i32_data(c4_sparse_topk_lengths);
    auto *c4_out_loc_ptr = i32_data(c4_out_loc);
    auto *c4_positions_ptr = i32_data(c4_positions);
    auto *c4_compress_write_loc_ptr = i32_data(c4_compress_write_loc);
    auto *c4_compress_extra_loc_ptr = i32_data(c4_compress_extra_loc);
    auto *c128_page_indices_ptr = i32_data(c128_page_indices);
    auto *c128_topk_lengths_clamp1_ptr = i32_data(c128_topk_lengths_clamp1);
    auto *c128_out_loc_ptr = i32_data(c128_out_loc);
    auto *c128_positions_ptr = i32_data(c128_positions);
    auto *c128_compress_write_loc_ptr = i32_data(c128_compress_write_loc);
    std::fill(c128_page_indices_ptr, c128_page_indices_ptr + rows * c128_width, -1);

    for (size_t req = 0; req < req_count; ++req) {
        const int32_t begin = input_offsets_ptr[req];
        const int32_t end = input_offsets_ptr[req + 1];
        if (begin < 0 || end < begin || static_cast<size_t>(end) > rows) {
            throw std::runtime_error("build_deepseek_v4_attention_metadata: invalid input_offsets");
        }
        const int32_t *block_table = block_tables_ptr + req * max_block_table_len;
        if (begin == end) {
            continue;
        }

        const size_t req_rows = static_cast<size_t>(end - begin);
        if (req_rows < 16) {
            for (size_t row = static_cast<size_t>(begin); row < static_cast<size_t>(end); ++row) {
                const int64_t position = read_position(position_ids_ptr, position_ids->dtype(), row);
                const int64_t slot = slot_mapping_ptr[row];

                std::memcpy(page_table_ptr + row * max_block_table_len,
                            block_table,
                            max_block_table_len * sizeof(int32_t));

                raw_out_loc_ptr[row] = checked_i32(full_slot_to_swa_slot(slot, full_to_swa_block_ids, block_size),
                                                   "raw_out_loc");

                for (size_t offset = 0; offset < swa_window_size; ++offset) {
                    const int64_t full_slot = slot_for_position(
                        block_table, max_block_table_len, position - static_cast<int64_t>(offset), block_size);
                    swa_indices_ptr[row * swa_window_size + offset] =
                        checked_i32(full_slot_to_swa_slot(full_slot, full_to_swa_block_ids, block_size),
                                    "swa_indices");
                }
                swa_topk_lengths_ptr[row] = checked_i32(std::min<int64_t>(position + 1, 128), "swa_topk_lengths");

                const int64_t c4_visible = (position + 1) / 4;
                const int64_t c4_clamp1 = std::max<int64_t>(c4_visible, 1);
                c4_topk_lengths_raw_ptr[row] = checked_i32(c4_visible, "c4_topk_lengths_raw");
                c4_sparse_topk_lengths_ptr[row] = checked_i32(std::min<int64_t>(c4_clamp1, c4_sparse_topk),
                                                              "c4_sparse_topk_lengths");
                const auto c4_loc = compress_locations_for_position(
                    block_table, max_block_table_len, slot, position, 4, block_size, full_to_swa_block_ids);
                c4_out_loc_ptr[row] = checked_i32(c4_loc.out_loc, "c4_out_loc");
                c4_positions_ptr[row] = checked_i32(c4_loc.clipped_position, "c4_positions");
                c4_compress_write_loc_ptr[row] = checked_i32(c4_loc.write_loc, "c4_compress_write_loc");
                c4_compress_extra_loc_ptr[row] = checked_i32(c4_loc.prev_state_index, "c4_compress_extra_loc");

                const int64_t c128_visible = (position + 1) / 128;
                const int64_t c128_clamp1 = std::max<int64_t>(c128_visible, 1);
                c128_topk_lengths_clamp1_ptr[row] = checked_i32(c128_clamp1, "c128_topk_lengths_clamp1");
                for (size_t compressed_idx = 0; compressed_idx < c128_width; ++compressed_idx) {
                    if (static_cast<int64_t>(compressed_idx) >= c128_visible) {
                        c128_page_indices_ptr[row * c128_width + compressed_idx] = -1;
                        continue;
                    }
                    const int64_t raw_position = (static_cast<int64_t>(compressed_idx) + 1) * 128 - 1;
                    const int64_t raw_slot = slot_for_position(block_table, max_block_table_len, raw_position, block_size);
                    c128_page_indices_ptr[row * c128_width + compressed_idx] =
                        checked_i32(raw_slot >= 0 ? raw_slot / 128 : -1, "c128_page_indices");
                }

                const auto c128_loc = compress_locations_for_position(
                    block_table, max_block_table_len, slot, position, 128, block_size, full_to_swa_block_ids);
                c128_out_loc_ptr[row] = checked_i32(c128_loc.out_loc, "c128_out_loc");
                c128_positions_ptr[row] = checked_i32(c128_loc.clipped_position, "c128_positions");
                c128_compress_write_loc_ptr[row] = checked_i32(c128_loc.write_loc, "c128_compress_write_loc");
            }
            continue;
        }

        size_t max_req_position = 0;
        for (size_t row = static_cast<size_t>(begin); row < static_cast<size_t>(end); ++row) {
            const int64_t position = read_position(position_ids_ptr, position_ids->dtype(), row);
            if (position < 0) {
                throw std::runtime_error("build_deepseek_v4_attention_metadata: negative position_id");
            }
            max_req_position = std::max(max_req_position, static_cast<size_t>(position));
        }
        const auto mapped_slots = build_mapped_slots_by_position(
            block_table, max_block_table_len, max_req_position, block_size, full_to_swa_block_ids);
        const auto c4_state_locs = build_state_locs_by_position(mapped_slots, 4, block_size);
        const auto c128_state_locs = build_state_locs_by_position(mapped_slots, 128, block_size);

        std::vector<int32_t> c128_page_by_index(c128_width, -1);
        for (size_t compressed_idx = 0; compressed_idx < c128_width; ++compressed_idx) {
            const int64_t raw_position = (static_cast<int64_t>(compressed_idx) + 1) * 128 - 1;
            const int64_t raw_slot = slot_for_position(block_table, max_block_table_len, raw_position, block_size);
            c128_page_by_index[compressed_idx] =
                checked_i32(raw_slot >= 0 ? raw_slot / 128 : -1, "c128_page_indices");
        }

        for (size_t row = static_cast<size_t>(begin); row < static_cast<size_t>(end); ++row) {
            const int64_t position = read_position(position_ids_ptr, position_ids->dtype(), row);
            const int64_t slot = slot_mapping_ptr[row];

            std::memcpy(page_table_ptr + row * max_block_table_len,
                        block_table,
                        max_block_table_len * sizeof(int32_t));

            raw_out_loc_ptr[row] = checked_i32(full_slot_to_swa_slot(slot, full_to_swa_block_ids, block_size),
                                               "raw_out_loc");

            auto *swa_row = swa_indices_ptr + row * swa_window_size;
            const size_t swa_len = std::min<size_t>(static_cast<size_t>(position + 1), swa_window_size);
            for (size_t offset = 0; offset < swa_len; ++offset) {
                swa_row[offset] = mapped_slot_for_position(
                    mapped_slots,
                    block_table,
                    max_block_table_len,
                    position - static_cast<int64_t>(offset),
                    block_size,
                    full_to_swa_block_ids);
            }
            if (swa_len < swa_window_size) {
                std::fill(swa_row + swa_len, swa_row + swa_window_size, -1);
            }
            swa_topk_lengths_ptr[row] = checked_i32(std::min<int64_t>(position + 1, 128), "swa_topk_lengths");

            const int64_t c4_visible = (position + 1) / 4;
            const int64_t c4_clamp1 = std::max<int64_t>(c4_visible, 1);
            c4_topk_lengths_raw_ptr[row] = checked_i32(c4_visible, "c4_topk_lengths_raw");
            c4_sparse_topk_lengths_ptr[row] = checked_i32(std::min<int64_t>(c4_clamp1, c4_sparse_topk),
                                                          "c4_sparse_topk_lengths");

            const int64_t c4_clipped_position = (position / 4) * 4;
            const int64_t c4_state_slot = full_slot_to_swa_slot(slot, full_to_swa_block_ids, block_size);
            const int32_t c4_state_index = checked_i32(state_loc(c4_state_slot, 4, block_size), "c4_state_index");
            int32_t c4_write = state_loc_for_mapped_position(
                c4_state_locs,
                mapped_slots,
                block_table,
                max_block_table_len,
                c4_clipped_position,
                4,
                block_size,
                full_to_swa_block_ids);
            if (c4_write < 0) {
                c4_write = c4_state_index;
            }
            int32_t c4_prev = state_loc_for_mapped_position(
                c4_state_locs,
                mapped_slots,
                block_table,
                max_block_table_len,
                c4_clipped_position - 4,
                4,
                block_size,
                full_to_swa_block_ids);
            if (c4_prev < 0) {
                c4_prev = c4_write >= 0 ? c4_write : 0;
            }
            c4_out_loc_ptr[row] = checked_i32(slot >= 0 && (position + 1) % 4 == 0 ? slot / 4 : -1, "c4_out_loc");
            c4_positions_ptr[row] = checked_i32(c4_clipped_position, "c4_positions");
            c4_compress_write_loc_ptr[row] = c4_write;
            c4_compress_extra_loc_ptr[row] = c4_prev;

            const int64_t c128_visible = (position + 1) / 128;
            const int64_t c128_clamp1 = std::max<int64_t>(c128_visible, 1);
            c128_topk_lengths_clamp1_ptr[row] = checked_i32(c128_clamp1, "c128_topk_lengths_clamp1");
            if (c128_visible > 0) {
                std::memcpy(c128_page_indices_ptr + row * c128_width,
                            c128_page_by_index.data(),
                            std::min<size_t>(static_cast<size_t>(c128_visible), c128_width) * sizeof(int32_t));
            }

            const int64_t c128_clipped_position = (position / 128) * 128;
            const int64_t c128_state_slot = full_slot_to_swa_slot(slot, full_to_swa_block_ids, block_size);
            const int32_t c128_state_index = checked_i32(state_loc(c128_state_slot, 128, block_size), "c128_state_index");
            int32_t c128_write = state_loc_for_mapped_position(
                c128_state_locs,
                mapped_slots,
                block_table,
                max_block_table_len,
                c128_clipped_position,
                128,
                block_size,
                full_to_swa_block_ids);
            if (c128_write < 0) {
                c128_write = c128_state_index;
            }
            c128_out_loc_ptr[row] = checked_i32(slot >= 0 && (position + 1) % 128 == 0 ? slot / 128 : -1,
                                                "c128_out_loc");
            c128_positions_ptr[row] = checked_i32(c128_clipped_position, "c128_positions");
            c128_compress_write_loc_ptr[row] = c128_write;
        }
    }

    return {
        swa_indices,
        swa_topk_lengths,
        raw_out_loc,
        page_table,
        c4_out_loc,
        c4_positions,
        c4_topk_lengths_raw,
        c4_sparse_topk_lengths,
        c128_out_loc,
        c128_positions,
        c128_page_indices,
        c128_topk_lengths_clamp1,
        c4_compress_write_loc,
        c4_compress_extra_loc,
        c128_compress_write_loc,
    };
}

} // namespace infinilm::engine
