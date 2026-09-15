#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>

namespace infinilm::layers::mla_attention {

class Cache {
public:
    virtual ~Cache() = default;

    virtual size_t page_size() const noexcept = 0;
    virtual const infinicore::Tensor &raw_cache_view() const noexcept = 0;
    virtual const infinicore::Tensor &flashmla_cache_view() const noexcept = 0;
};

class DenseFlashMLACache final : public Cache {
public:
    DenseFlashMLACache(size_t num_blocks,
                       size_t block_size,
                       size_t kv_lora_rank,
                       size_t qk_rope_head_dim,
                       infinicore::DataType dtype,
                       const infinicore::Device &device);

    size_t page_size() const noexcept override;
    const infinicore::Tensor &raw_cache_view() const noexcept override;
    const infinicore::Tensor &flashmla_cache_view() const noexcept override;
    void set_key_buffer(const infinicore::Tensor &kv_c,
                        const infinicore::Tensor &k_pe,
                        const infinicore::Tensor &slot_mapping);

private:
    infinicore::Tensor cache_;
};

class SparseFlashMLACache final : public Cache {
public:
    SparseFlashMLACache(size_t num_blocks,
                        size_t block_size,
                        size_t qk_nope_head_dim,
                        size_t qk_rope_head_dim,
                        infinicore::DataType dtype,
                        const infinicore::Device &device);

    size_t get_bytes_per_token() const noexcept;

    size_t page_size() const noexcept override;
    const infinicore::Tensor &raw_cache_view() const noexcept override;
    const infinicore::Tensor &flashmla_cache_view() const noexcept override;
    void set_key_buffer(const infinicore::Tensor &cache_k,
                        const infinicore::Tensor &loc);

private:
    size_t qk_nope_head_dim_;
    size_t qk_rope_head_dim_;
    size_t scale_pad_{1};
    size_t quantize_block_size_{64};
    infinicore::DataType rope_storage_dtype_{infinicore::DataType::BF16};
    infinicore::DataType k_with_scale_buffer_dtype_{infinicore::DataType::U8};

    infinicore::Tensor raw_cache_;
    infinicore::Tensor flashmla_cache_view_;
};

} // namespace infinilm::layers::mla_attention
