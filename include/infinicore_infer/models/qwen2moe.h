#ifndef _QWEN2MOE_H_
#define _QWEN2MOE_H_

#include "qwen3moe.h"
#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>
#include <stdint.h>
#include <stdio.h>

namespace Qwen2MoE {
struct Weights;
struct Model;

using Meta = Qwen3MoE::Meta;
}; // namespace Qwen2MoE

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////           Qwen3 APIs            /////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct Qwen2MoE::Model *
Qwen2MoEcreateModel(const Qwen2MoE::Meta *,
                    const Qwen2MoE::Weights *,
                    infiniDevice_t device,
                    int ndev,
                    const int *dev_ids);

/// @brief 销毁模型
__C __export void
Qwen2MoEdestroyModel(struct Qwen2MoE::Model *);

/// @brief 创建 KV Cache
__C __export struct KVCache *
Qwen2MoEcreateKVCache(const struct Qwen2MoE::Model *);

/// @brief 复制 KV Cache
__C __export struct KVCache *
Qwen2MoEduplicateKVCache(const struct Qwen2MoE::Model *,
                         const struct KVCache *, uint32_t seq_len);

/// @brief 销毁 KV Cache
__C __export void
Qwen2MoEdropKVCache(const struct Qwen2MoE::Model *,
                    struct KVCache *);

/// @brief 批次推理一轮，并采样出新的 token
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param temperature 采样温度（0. 表示贪心采样）
/// @param topk 采样 topk（1 表示贪心采样）
/// @param topp 采样 topp
/// @param output 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
Qwen2MoEinferBatch(struct Qwen2MoE::Model *,
                   const uint32_t *tokens, uint32_t ntok,
                   const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                   struct KVCache **kv_caches,
                   const float *temperature, const uint32_t *topk, const float *topp,
                   uint32_t *output);

/// @brief 批次推理一轮，输出 output embedding 后的 logits
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param logits 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
Qwen2MoEforwardBatch(struct Qwen2MoE::Model *,
                     const uint32_t *tokens, uint32_t ntok,
                     const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                     struct KVCache **kv_caches,
                     void *logits);

#endif
