#include "deepseek_v4_for_causal_lm.hpp"

#include "../../cache/kv_cache.hpp"
#include "../../global_state/global_state.hpp"

#include "infinicore/context/context.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {

namespace {

constexpr size_t kDsv4FlashMlaValueBytesPerToken = 576;
constexpr size_t kDsv4FlashMlaScaleBytesPerToken = 8;
constexpr size_t kDsv4SwaPageSize = 256;
constexpr size_t kDsv4C4PageSize = 64;
constexpr size_t kDsv4C128PageSize = 2;
constexpr size_t kDsv4IndexerQuantBlockSize = 128;
constexpr size_t kDsv4IndexerScaleBytesPerBlock = sizeof(float);

constexpr size_t round_up_to_multiple(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

constexpr size_t flashmla_raw_cache_page_bytes(size_t page_size) {
    // FlashMLA raw cache pages are padded to the value-area byte alignment.
    return round_up_to_multiple(
        (kDsv4FlashMlaValueBytesPerToken + kDsv4FlashMlaScaleBytesPerToken) * page_size,
        kDsv4FlashMlaValueBytesPerToken);
}

} // namespace

DeepseekV4KVCacheTensors deepseek_v4_allocate_kv_cache_tensors(const cache::CacheConfig *cache_config,
                                                               const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
                                                               const backends::AttentionBackend &attention_backend) {
    if (cache_config == nullptr) {
        return {};
    }
    if (text_config == nullptr) {
        throw std::runtime_error("infinilm::models::deepseek_v4::deepseek_v4_allocate_kv_cache_tensors: text_config is null");
    }

    if (attention_backend != backends::AttentionBackend::PAGED_ATTN) {
        throw std::runtime_error("infinilm::models::deepseek_v4::deepseek_v4_allocate_kv_cache_tensors: DeepSeek V4 requires paged attention backend");
    }

    auto paged_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
    if (paged_cache_config == nullptr) {
        throw std::runtime_error("infinilm::models::deepseek_v4::deepseek_v4_allocate_kv_cache_tensors: invalid paged kv cache config type");
    }

    DeepseekV4KVCacheTensors cache_tensors;
    const auto device = infinilm::global_state::get_tensor_model_parallel_rank_info().device;
    const size_t num_blocks = paged_cache_config->num_blocks();
    const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
    const size_t head_dim = text_config->get<size_t>("head_dim");
    const size_t index_head_dim = text_config->get_or<size_t>("index_head_dim", 128);
    const auto compress_ratios = text_config->get<std::vector<size_t>>("compress_ratios");
    const size_t swa_num_blocks = std::max<size_t>(1, num_blocks / 10);
    if (index_head_dim % kDsv4IndexerQuantBlockSize != 0) {
        throw std::runtime_error("infinilm::models::deepseek_v4::deepseek_v4_allocate_kv_cache_tensors: index_head_dim must be divisible by 128");
    }
    const size_t swa_page_bytes = flashmla_raw_cache_page_bytes(kDsv4SwaPageSize);
    const size_t c4_page_bytes = flashmla_raw_cache_page_bytes(kDsv4C4PageSize);
    const size_t c128_page_bytes = flashmla_raw_cache_page_bytes(kDsv4C128PageSize);
    const size_t indexer_page_bytes = kDsv4C4PageSize * (index_head_dim + (index_head_dim / kDsv4IndexerQuantBlockSize) * kDsv4IndexerScaleBytesPerBlock);

    cache_tensors.kv_cache_tensors.reserve(num_hidden_layers);
    cache_tensors.deepseek_v4_kv_cache_tensors.reserve(num_hidden_layers);
    for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
        const size_t compress_ratio = layer_idx < compress_ratios.size() ? compress_ratios[layer_idx] : 0;
        infinilm::global_state::DeepSeekV4LayerKVCache layer_cache;
        layer_cache.swa_cache_raw = infinicore::Tensor::zeros({swa_num_blocks, swa_page_bytes}, infinicore::DataType::U8, device);
        layer_cache.c4_cache_raw = infinicore::Tensor::zeros({num_blocks, c4_page_bytes}, infinicore::DataType::U8, device);
        if (compress_ratio == 4) {
            layer_cache.c4_indexer_cache_raw = infinicore::Tensor::zeros({num_blocks, indexer_page_bytes}, infinicore::DataType::U8, device);
        }
        layer_cache.c128_cache_raw = infinicore::Tensor::zeros({num_blocks, c128_page_bytes}, infinicore::DataType::U8, device);
        layer_cache.kv_scale = infinicore::Tensor::ones({1}, infinicore::DataType::F32, device);

        if (compress_ratio != 0) {
            const size_t coeff = compress_ratio == 4 ? 2 : 1;
            const size_t ring_size = compress_ratio == 4 ? 8 : 128;
            size_t state_rows = swa_num_blocks * ring_size + ring_size + 1;
            state_rows = ((state_rows + compress_ratio - 1) / compress_ratio) * compress_ratio;
            layer_cache.compressor_state = infinicore::Tensor::zeros({state_rows, 2 * coeff * head_dim}, infinicore::DataType::F32, device);
            if (compress_ratio == 4) {
                layer_cache.indexer_compressor_state = infinicore::Tensor::zeros({state_rows, 2 * coeff * index_head_dim}, infinicore::DataType::F32, device);
            }
        }

        // Expose the primary SWA cache through the legacy getter while keeping
        // the full DSv4 cache tuple in the dedicated forward context field.
        cache_tensors.kv_cache_tensors.push_back(layer_cache.swa_cache_raw);
        cache_tensors.deepseek_v4_kv_cache_tensors.push_back(std::move(layer_cache));
    }
    infinicore::context::syncStream();
    return cache_tensors;
}

} // namespace infinilm::models::deepseek_v4
