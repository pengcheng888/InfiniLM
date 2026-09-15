#include "flashmla_cache.hpp"

#include "../../utils.hpp"

#include "infinicore/ops/concat_and_cache_mla.hpp"
#include "infinicore/ops/deepseek_v4/fused_store_flashmla_cache.hpp"

#include <stdexcept>

namespace infinilm::layers::mla_attention {
namespace {

size_t round_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

} // namespace

DenseFlashMLACache::DenseFlashMLACache(
    size_t num_blocks,
    size_t block_size,
    size_t kv_lora_rank,
    size_t qk_rope_head_dim,
    infinicore::DataType dtype,
    const infinicore::Device &device) {
    if (num_blocks == 0 || block_size == 0) {
        throw std::runtime_error("DenseFlashMLACache expects positive num_blocks and block_size");
    }
    if (kv_lora_rank == 0 || qk_rope_head_dim == 0) {
        throw std::runtime_error("DenseFlashMLACache expects positive kv_lora_rank and qk_rope_head_dim");
    }

    const size_t cache_dim = kv_lora_rank + qk_rope_head_dim;
    cache_ = infinicore::Tensor::empty({num_blocks, block_size, 1, cache_dim}, dtype, device);
}

size_t DenseFlashMLACache::page_size() const noexcept {
    return cache_->size(1);
}

const infinicore::Tensor &DenseFlashMLACache::raw_cache_view() const noexcept {
    return cache_;
}

const infinicore::Tensor &DenseFlashMLACache::flashmla_cache_view() const noexcept {
    return cache_;
}

void DenseFlashMLACache::set_key_buffer(const infinicore::Tensor &kv_c,
                                        const infinicore::Tensor &k_pe,
                                        const infinicore::Tensor &slot_mapping) {
    const auto &raw_cache = raw_cache_view();
    if (raw_cache->ndim() != 4 || raw_cache->size(2) != 1) {
        throw std::runtime_error(
            "DenseFlashMLACache::set_key_buffer expects KV cache [blocks, block_size, 1, head_size]");
    }

    auto kv_cache_3d = raw_cache->view(
        {raw_cache->size(0), raw_cache->size(1), raw_cache->size(3)});
    auto cache_scale = infinicore::Tensor::ones(
        {1}, infinicore::DataType::F32, raw_cache->device());
    infinicore::op::concat_and_cache_mla_(
        kv_c, k_pe, kv_cache_3d, slot_mapping, "auto", cache_scale);
}

SparseFlashMLACache::SparseFlashMLACache(
    size_t num_blocks,
    size_t block_size,
    size_t qk_nope_head_dim,
    size_t qk_rope_head_dim,
    infinicore::DataType dtype,
    const infinicore::Device &device)
    : qk_nope_head_dim_(qk_nope_head_dim),
      qk_rope_head_dim_(qk_rope_head_dim) {
    if (num_blocks == 0 || block_size == 0) {
        throw std::runtime_error("SparseFlashMLACache expects positive num_blocks and block_size");
    }
    if ((block_size & (block_size - 1)) != 0) {
        throw std::runtime_error("SparseFlashMLACache expects block_size to be a power of two");
    }
    if (qk_nope_head_dim_ == 0 || qk_rope_head_dim_ == 0) {
        throw std::runtime_error("SparseFlashMLACache expects positive qk_nope_head_dim and qk_rope_head_dim");
    }
    if (dtype != k_with_scale_buffer_dtype_) {
        throw std::runtime_error("SparseFlashMLACache requires U8 raw cache storage");
    }
    if (qk_nope_head_dim_ % quantize_block_size_ != 0) {
        throw std::runtime_error("SparseFlashMLACache expects qk_nope_head_dim to be divisible by quantize_block_size");
    }

    const size_t value_bytes_per_token = qk_nope_head_dim_ * infinicore::dsize(k_with_scale_buffer_dtype_)
                                       + qk_rope_head_dim_ * infinicore::dsize(rope_storage_dtype_);
    const size_t bytes_per_token = get_bytes_per_token(); // 584

    ASSERT(bytes_per_token == (448 + 64 * 2 + 8));
    ASSERT(dtype == infinicore::DataType::U8);

    const size_t bytes_per_page_padded = round_up(block_size * bytes_per_token, value_bytes_per_token);

    raw_cache_ = infinicore::Tensor::zeros({num_blocks, bytes_per_page_padded}, k_with_scale_buffer_dtype_, device);
    auto raw_cache_f8 = infinicore::Tensor::from_blob(raw_cache_->data(), {num_blocks, bytes_per_page_padded}, infinicore::DataType::F8, device);

    flashmla_cache_view_ = raw_cache_f8->narrow({{1, 0, block_size * bytes_per_token}})
                               ->view({num_blocks, block_size, 1, bytes_per_token});
}

size_t SparseFlashMLACache::get_bytes_per_token() const noexcept {
    const size_t value_bytes_per_token = qk_nope_head_dim_ * infinicore::dsize(k_with_scale_buffer_dtype_)
                                       + qk_rope_head_dim_ * infinicore::dsize(rope_storage_dtype_);
    const size_t scale_bytes_per_token = qk_nope_head_dim_ / quantize_block_size_ + scale_pad_;
    return value_bytes_per_token + scale_bytes_per_token;
}

size_t SparseFlashMLACache::page_size() const noexcept {
    return flashmla_cache_view_->size(1);
}

const infinicore::Tensor &SparseFlashMLACache::raw_cache_view() const noexcept {
    return raw_cache_;
}

const infinicore::Tensor &SparseFlashMLACache::flashmla_cache_view() const noexcept {
    return flashmla_cache_view_;
}

void SparseFlashMLACache::set_key_buffer(const infinicore::Tensor &cache_k,
                                         const infinicore::Tensor &loc) {
    infinicore::op::deepseek_v4::fused_store_flashmla_cache_(
        cache_k, raw_cache_view(), loc, static_cast<int>(page_size()));
}

} // namespace infinilm::layers::mla_attention
