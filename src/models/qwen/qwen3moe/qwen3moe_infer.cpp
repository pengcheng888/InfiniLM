#include "../../../tensor.hpp"
#include "../../../utils.hpp"
#include "../../inference_context.hpp"
#include "../qwen_device_resource.hpp"
#include "../qwen_model.hpp"
#include "infinicore_infer.h"
#include <random>
#include <thread>
#include <vector>

void Qwen3MoEinferDeviceBatch(const Qwen3MoE::Meta *meta, DeviceResource &rsrc,
                              uint32_t idev, uint32_t ndev,
                              const uint32_t *tokens, uint32_t ntok,
                              const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                              struct KVCache **kv_caches,
                              const float *temperature, const uint32_t *topk, const float *topp,
                              uint32_t *output, void *last_logits) {

    auto nlayer = meta->nlayer;
    auto nkvh = meta->nkvh / ndev;
    auto nh = meta->nh / ndev;
    auto ngroup = nh / nkvh;
    // auto dctx = meta.dctx;
    auto dh = meta->dh;
    auto d = meta->d;
    auto dt_logits = meta->dt_logits;
    // auto di = meta->di / ndev;
    auto dvoc = meta->dvoc;
    auto stream = rsrc.stream;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);

    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});

    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    const Qwen3MoE::WeightsTensor *g_WeightsTensor{nullptr};
    if (rsrc.weights_tensor_ptr) {
        g_WeightsTensor = static_cast<const Qwen3MoE::WeightsTensor *>(rsrc.weights_tensor_ptr);
    } else {
        return;
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d), g_WeightsTensor->w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // Attention
    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }

    auto qk_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
    auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});

    // Compute
    for (uint32_t ilayer = 0; ilayer < nlayer; ilayer++) {
        auto layer_tensor = g_WeightsTensor->layers[ilayer];

        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, layer_tensor->w_attn_norm, meta->epsilon);
        // qkv_proj
        linear(qkv_buf, logits_out, layer_tensor->self_attn->w_attn_qkv, 1.0, 0.0, nullptr, layer_tensor->self_attn->b_attn_qkv ? layer_tensor->self_attn->b_attn_qkv : nullptr);

        if (layer_tensor->self_attn->w_attn_qk_norm) {
            auto qkv_buf_view = qkv_buf->view({ntok, nh + nkvh * 2, dh});
            auto q_buf = qkv_buf_view->slice(1, 0, nh);
            auto k_buf = qkv_buf_view->slice(1, nh, nkvh);
            rmsnorm(q_buf, q_buf, layer_tensor->self_attn->w_attn_qk_norm->slice(0, 0, dh), meta->epsilon);
            rmsnorm(k_buf, k_buf, layer_tensor->self_attn->w_attn_qk_norm->slice(0, dh, dh), meta->epsilon);
        }

        // rope
        rope(qkv_rope->slice(1, 0, nh), qkv_rope->slice(1, 0, nh), pos_ids_buf, g_WeightsTensor->sin_table, g_WeightsTensor->cos_table);
        rope(qkv_rope->slice(1, nh, nkvh), qkv_rope->slice(1, nh, nkvh), pos_ids_buf, g_WeightsTensor->sin_table, g_WeightsTensor->cos_table);

        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto total_len = past_len + seq_len;
            auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto q = qkv_rope->slice({{0, token_offset, seq_len}, {1, 0, nh}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
            auto v = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});

            // self attention
            // concat
            rearrange(kv_caches[req]->k[idev][ilayer]->slice(0, past_len, seq_len), k);
            rearrange(kv_caches[req]->v[idev][ilayer]->slice(0, past_len, seq_len), v);
            // qk
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(1, 0, seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
            auto k_gemm = kv_caches[req]->k[idev][ilayer]->slice(0, 0, total_len)->permute({1, 2, 0});
            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = qk_buf->slice(1, 0, seq_len * total_len)->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);
            auto v_gemm = kv_caches[req]->v[idev][ilayer]->slice(0, 0, total_len)->permute({1, 0, 2});
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
            // rearrange attn val
            rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

            token_offset += seq_len;
        }

        // o_proj
        linear(logits_in, o_buf, layer_tensor->self_attn->w_attn_out, 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }

        // 2. FFN
        rmsnorm(logits_out, logits_in, layer_tensor->w_ffn_norm, meta->epsilon);

        // ------------------------------------------------------------------------ //
        //                          SparseMLP                                       //
        // ------------------------------------------------------------------------ //
        {
            // 明确输入输出变量
            std::shared_ptr<Tensor> hidden_states = logits_out; // logits_out 是整个 MoE的输入，重新起名字为 hidden_states

            // 需要提前申请的缓存
            size_t moe_intermediate_size = meta->_moe_intermediate_size / ndev;

            // 需要提前申请的缓存
            auto router_gate_up_buf = Tensor::buffer(dt_logits, {1, 2 * moe_intermediate_size}, rsrc.memory_pool);
            auto router_gate_buf = router_gate_up_buf->slice(1, 0, moe_intermediate_size);
            auto router_up_buf = router_gate_up_buf->slice(1, moe_intermediate_size,moe_intermediate_size);

            // 需要提前申请的缓存
            std::shared_ptr<Tensor> router_states_sum = Tensor::buffer(hidden_states->dtype(), hidden_states->shape(), rsrc.memory_pool); // 用于存储 SparseMLP的输出

            // 需要提前申请的缓存
            std::shared_ptr<Tensor> router_logits = Tensor::buffer(dt_logits, {ntok, meta->_num_experts}, rsrc.memory_pool); // 路由专家的权重

            //
            size_t topk = meta->_num_experts_per_tok;
            bool norm_topk_prob = meta->_norm_topk_prob;

            std::shared_ptr<Tensor> values_gpu = Tensor::buffer(infiniDtype_t::INFINI_DTYPE_F32, {ntok * topk}, rsrc.memory_pool);  // 用于存储topkrouter的输出，每个expert对应的加权权重。
            std::shared_ptr<Tensor> indices_gpu = Tensor::buffer(infiniDtype_t::INFINI_DTYPE_I32, {ntok * topk}, rsrc.memory_pool); // 用于存储topkrouter的输出，要经过哪些专家id（从256个中选8个）
            std::vector<float> values_cpu(ntok * topk, 0.f);                                                                        // 用于存储topkrouter的输出，每个expert对应的加权权重。（从256个中选8个）
            std::vector<int> indices_cpu(ntok * topk, 0);                                                                           // 用于存储topkrouter的输出，要经过哪些专家的索引。

            // ------------------------------------------------------------------------ //
            //                            开始计算                                       //
            // ------------------------------------------------------------------------ //
            auto ffn = layer_tensor->ffn;

            // (1) topk操作：
            //      hidden_states 先经过 gate_weight，得到 router_logits
            //      router_logits 进行 topk 操作
            linear(router_logits, hidden_states, ffn->_gate_weight, 1.0, 0.0, nullptr, nullptr); // 这一行的代码是正确的
            {
                topksoftmax(values_gpu, indices_gpu, router_logits, topk, norm_topk_prob);
                RUN_INFINI(infinirtMemcpy((void *)values_cpu.data(), values_gpu->data(), values_cpu.size() * sizeof(float), INFINIRT_MEMCPY_D2H));
                RUN_INFINI(infinirtMemcpy((void *)indices_cpu.data(), indices_gpu->data(), indices_cpu.size() * sizeof(int), INFINIRT_MEMCPY_D2H));
                RUN_INFINI(infinirtStreamSynchronize(rsrc.stream));
            }

            // (2) MoE操作：  每个 token 经过 4个 路由专家
            {
                // 输入: hidden_states
                //      values_cpu，indices_cpu
                for (size_t itok = 0; itok < ntok; ++itok) {
                    std::shared_ptr<Tensor> hidden_states_i = hidden_states->slice(0, itok, 1);
                    std::shared_ptr<Tensor> router_states_sum_i = router_states_sum->slice(0, itok, 1);

                    // 经过第一个专家 : C = alpha * AB
                    {
                        int index = indices_cpu[itok * topk + 0];
                        float alpha = values_cpu[itok * topk + 0];
                        linear(router_gate_up_buf, hidden_states_i, layer_tensor->ffn->_experts[index]->w_ffn_gate_up, 1.0, 0.0, nullptr, nullptr);
                        swiglu(router_gate_buf, router_up_buf, router_gate_buf);
                        linear(router_states_sum_i, router_gate_buf, layer_tensor->ffn->_experts[index]->w_ffn_down, alpha, 0.0, nullptr, nullptr);
                    }

                    // 经过后续的专家 : C  = alpha * AB + C_last
                    for (size_t k = 1; k < topk; ++k) {
                        int index = indices_cpu[itok * topk + k];
                        float alpha = values_cpu[itok * topk + k];
                        linear(router_gate_up_buf, hidden_states_i, layer_tensor->ffn->_experts[index]->w_ffn_gate_up, 1.0, 0.0, nullptr, nullptr);
                        swiglu(router_gate_buf, router_up_buf, router_gate_buf);
                        linear(router_states_sum_i, router_gate_buf, layer_tensor->ffn->_experts[index]->w_ffn_down, alpha, 0.0, router_states_sum_i, nullptr);
                    }
                }

                if (rsrc.comm != nullptr) {
                    RUN_INFINI(infinicclAllReduce(
                        router_states_sum->data(), router_states_sum->data(), ntok * d, dt_logits,
                        INFINICCL_SUM, rsrc.comm, stream));
                    RUN_INFINI(infinirtStreamSynchronize(stream));
                }
            }

            // (3) 最后的残差连接
            add(logits_in, router_states_sum, logits_in);
        }

        // All_reduce if distributed
        // if (rsrc.comm != nullptr) {
        //     RUN_INFINI(infinicclAllReduce(
        //         logits_in->data(), logits_in->data(), ntok * d, dt_logits,
        //         INFINICCL_SUM, rsrc.comm, stream));
        //     RUN_INFINI(infinirtStreamSynchronize(stream));
        // }
    }

    // Sample and Output
    if (idev == 0) {
        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, g_WeightsTensor->w_out_norm, meta->epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, g_WeightsTensor->w_out_embd, 1.0, 0.0, nullptr, nullptr);
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(last_logits, last_logits_buf->data(), dsize(dt_logits) * ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (output != nullptr) {
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, req, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        g_WeightsTensor->w_out_norm,
                        meta->epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, nreq), g_WeightsTensor->w_out_embd, 1.0, 0.0, nullptr, nullptr);
            std::random_device _rd;
            std::mt19937 gen(_rd());
            token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
                randomSample(result_buf->slice(0, req, 1)->view_as({}, {}),
                             prob_buf->slice(0, req, 1)->view_as({dvoc}, {1}),
                             random_val, topp[req], topk[req], temperature[req]);
                token_offset += seq_len;
            }
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                      sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
            for (uint32_t req = 0; req < nreq; req++) {
                output[req] = uint32_t(result_cpu[req]);
            }
        }
    }
}

namespace Qwen3MoE {
Model::Model(const Meta *_meta, const Weights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {

    printf("-------------> Model\n ");
    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<DeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice<WeightsTensor, Meta, Weights>, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i], Qwen3MoEinferDeviceBatch);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

}; // namespace Qwen3MoE
