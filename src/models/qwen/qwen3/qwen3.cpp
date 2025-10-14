#include "../../../tensor.hpp"
#include "../../../utils.hpp"
#include "../../inference_context.hpp"
#include "../qwen_device_resource.hpp"
#include "../qwen_kv_cache.hpp"
#include "../qwen_model.hpp"
#include "../qwen_weight.hpp"
#include "infinicore_infer.h"
#include <random>
#include <thread>
#include <vector>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////           Model API            ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

__C Qwen3::Model *Qwen3createModel(const Qwen3::Meta *meta,
                                   const Qwen3::Weights *weight,
                                   infiniDevice_t device,
                                   int ndev,
                                   const int *dev_ids) {
    return createModel<Qwen3::Model, Qwen3::Meta, Qwen3::Weights>(meta, weight, device, ndev, dev_ids);
}

/// @brief 销毁模型
__C void Qwen3destroyModel(struct Qwen3::Model *model) {
    destroyModel<Qwen3::Model>(model);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////           KVCache API            ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief 创建 KV Cache
__C KVCache *Qwen3createKVCache(const Qwen3::Model *model) {
    return createKVCache<Qwen3::Model>(model);
}

/// @brief 复制 KV Cache
__C KVCache *
Qwen3duplicateKVCache(const Qwen3::Model *model,
                      const KVCache *kv_cache, uint32_t seq_len) {
    return duplicateKVCache<Qwen3::Model>(model, kv_cache, seq_len);
}

/// @brief 销毁 KV Cache
__C void Qwen3dropKVCache(const Qwen3::Model *model, KVCache *kv_cache) {
    dropKVCache<Qwen3::Model>(model, kv_cache);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////           infer API            //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
__C void Qwen3inferBatch(struct Qwen3::Model *model,
                         const uint32_t *tokens, uint32_t ntok,
                         const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                         KVCache **kv_caches,
                         const float *temperature, const uint32_t *topk, const float *topp,
                         uint32_t *output) {
    inferBatch<Qwen3::Model>(model, tokens, ntok,
                             req_lens, nreq, req_pos,
                             kv_caches, temperature, topk, topp, output);
}

__C void Qwen3forwardBatch(Qwen3::Model *model,
                           const uint32_t *tokens, uint32_t ntok,
                           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                           KVCache **kv_caches,
                           void *logits) {

    forwardBatch<Qwen3::Model>(model,
                               tokens, ntok,
                               req_lens, nreq, req_pos,
                               kv_caches,
                               logits);
}