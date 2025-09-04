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

__C Qwen2MoE::Model *Qwen2MoEcreateModel(const Qwen2MoE::Meta *meta,
                                         const Qwen2MoE::Weights *weight,
                                         infiniDevice_t device,
                                         int ndev,
                                         const int *dev_ids) {
    return createModel<Qwen2MoE::Model, Qwen2MoE::Meta, Qwen2MoE::Weights>(meta, weight, device, ndev, dev_ids);
}

/// @brief 销毁模型
__C void Qwen2MoEdestroyModel(struct Qwen2MoE::Model *model) {
    destroyModel<Qwen2MoE::Model>(model);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////           KVCache API            ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief 创建 KV Cache
__C KVCache *Qwen2MoEcreateKVCache(const Qwen2MoE::Model *model) {
    return createKVCache<Qwen2MoE::Model>(model);
}

/// @brief 复制 KV Cache
__C KVCache *
Qwen2MoEduplicateKVCache(const Qwen2MoE::Model *model,
                         const KVCache *kv_cache, uint32_t seq_len) {
    return duplicateKVCache<Qwen2MoE::Model>(model, kv_cache, seq_len);
}

/// @brief 销毁 KV Cache
__C void Qwen2MoEdropKVCache(const Qwen2MoE::Model *model, KVCache *kv_cache) {
    dropKVCache<Qwen2MoE::Model>(model, kv_cache);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////           infer API            //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
__C void Qwen2MoEinferBatch(struct Qwen2MoE::Model *model,
                            const uint32_t *tokens, uint32_t ntok,
                            const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                            KVCache **kv_caches,
                            const float *temperature, const uint32_t *topk, const float *topp,
                            uint32_t *output) {
    inferBatch<Qwen2MoE::Model>(model, tokens, ntok,
                                req_lens, nreq, req_pos,
                                kv_caches, temperature, topk, topp, output);
}

__C void Qwen2MoEforwardBatch(Qwen2MoE::Model *model,
                              const uint32_t *tokens, uint32_t ntok,
                              const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                              KVCache **kv_caches,
                              void *logits) {

    forwardBatch<Qwen2MoE::Model>(model,
                                  tokens, ntok,
                                  req_lens, nreq, req_pos,
                                  kv_caches,
                                  logits);
}