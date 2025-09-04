#ifndef _QWEN3_H_
#define _QWEN3_H_

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>
#include <stdint.h>
#include <stdio.h>
namespace Qwen3 {
struct Weights;
struct Model;

struct Meta {
    infiniDtype_t dt_logits;
    size_t nlayer, d, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;

public:
    void print_info() const {
        printf(" dt_logits : %d\n", dt_logits);
        printf(" nlayer : %ld\n", nlayer);
        printf(" d : %ld\n", d);
        printf(" nh : %ld\n", nh);
        printf(" nkvh : %ld\n", nkvh);
        printf(" dh : %ld\n", dh);
        printf(" di : %ld\n", di);
        printf(" dvoc : %ld\n", dvoc);
        printf(" nkvh : %ld\n", nkvh);

        printf(" epsilon : %f\n", epsilon);
        printf(" theta : %f\n", theta);

        printf(" end_token : %d\n", end_token);
    }
};

}; // namespace Qwen3

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////           Qwen3 APIs            /////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct Qwen3::Model *
Qwen3createModel(const Qwen3::Meta *,
                 const Qwen3::Weights *,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids);

/// @brief 销毁模型
__C __export void
Qwen3destroyModel(struct Qwen3::Model *);

/// @brief 创建 KV Cache
__C __export struct KVCache *
Qwen3createKVCache(const struct Qwen3::Model *);

/// @brief 复制 KV Cache
__C __export struct KVCache *
Qwen3duplicateKVCache(const struct Qwen3::Model *,
                      const struct KVCache *, uint32_t seq_len);

/// @brief 销毁 KV Cache
__C __export void
Qwen3dropKVCache(const struct Qwen3::Model *,
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
Qwen3inferBatch(struct Qwen3::Model *,
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
Qwen3forwardBatch(struct Qwen3::Model *,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct KVCache **kv_caches,
                  void *logits);

#endif
