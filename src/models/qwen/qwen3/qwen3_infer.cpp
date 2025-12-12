#include "../../../tensor.hpp"
#include "../../../utils.hpp"
#include "../../inference_context.hpp"
#include "../qwen_device_resource.hpp"
#include "qwen3_model.hpp"
#include "qwen3_weight.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

void Qwen3inferDeviceBatch(const Qwen3::Meta *meta, DeviceResource<Qwen3::WeightsTensor> &rsrc,
                           uint32_t idev, uint32_t ndev,
                           const uint32_t *tokens, uint32_t ntok,
                           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                           struct KVCache **kv_caches,
                           const float *temperature, const uint32_t *topk, const float *topp,
                           uint32_t *output, void *last_logits) {
    auto nlayer = meta->nlayer;
    auto nkvh = meta->nkvh / ndev;
    auto nh = meta->nh / ndev;
    if (nkvh == 0) {
        throw std::invalid_argument("Qwen3inferDeviceBatch: nkvh / ndev is zero");
    }
    auto ngroup = nh / nkvh;
    if (ngroup == 0) {
        throw std::invalid_argument("Qwen3inferDeviceBatch: nh / nkvh is zero");
    }
    auto dh = meta->dh;
    auto d = meta->d;
    auto dt_logits = meta->dt_logits;
    auto di = meta->di / ndev;
    if (di == 0) {
        throw std::invalid_argument("Qwen3inferDeviceBatch: di / ndev is zero");
    }
    auto dvoc = meta->dvoc;
    auto stream = rsrc.stream;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);
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

    const Qwen3::WeightsTensor *g_WeightsTensor = rsrc.weights_tensor_ptr.get();
    if (!g_WeightsTensor) {
        throw std::runtime_error("Qwen3inferDeviceBatch: weights_tensor_ptr is nullptr");
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
        if (tokens[i] >= dvoc) {
            throw std::invalid_argument("Qwen3inferDeviceBatch: token index out of vocabulary range");
        }
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       g_WeightsTensor->w_in_embd->data(tokens[i] * d),
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

    // MLP buffers
    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);

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

            // qkv_rope shape: {ntok, nh + nkvh * 2, dh}
            auto q = qkv_rope->slice({{0, token_offset, seq_len}, {1, 0, nh}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
            auto v = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});

            // Update KV cache
            rearrange(kv_caches[req]->k[idev][ilayer]->slice(0, past_len, seq_len), k);
            rearrange(kv_caches[req]->v[idev][ilayer]->slice(0, past_len, seq_len), v);
            auto k_gemm = kv_caches[req]->k[idev][ilayer]->slice(0, 0, total_len)->permute({1, 2, 0}); //  {total_len, nkvh, dh} => { nkvh, dh, total_len}
            auto v_gemm = kv_caches[req]->v[idev][ilayer]->slice(0, 0, total_len)->permute({1, 0, 2}); //  {total_len, nkvh, dh} => { nkvh, total_len, dh}

            // self attention
            // qk
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(1, 0, seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});

            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = qk_buf->slice(1, 0, seq_len * total_len)->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);

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
        linear(gate_up_buf, logits_out, layer_tensor->ffn->w_ffn_gate_up, 1.0, 0.0, nullptr, nullptr);
        swiglu(gate_buf, up_buf, gate_buf);
        linear(logits_in, gate_buf, layer_tensor->ffn->w_ffn_down, 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
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

namespace Qwen3 {
/**
 * @brief Construct a Qwen3 Model instance.
 *
 * This constructor initializes the model with metadata, weights, and device configuration.
 * It sets up device resources, communication handles, and launches inference threads.
 *
 * @param _meta Pointer to model metadata (must not be nullptr).
 * @param weights Pointer to model weights, it is cpu pointer, it will be copied to gpu memory.
 * @param device_ The infiniDevice_t device type.
 * @param device_ids Vector of device IDs to use for inference.
 * @throws std::invalid_argument if _meta or weights is nullptr, or if device_ids is empty.
 * @throws std::runtime_error if device initialization fails.
 */
Model::Model(const Meta *_meta, const Weights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    // Input validation
    if (_meta == nullptr) {
        throw std::invalid_argument("Qwen3::Model::Model: _meta cannot be nullptr");
    }
    if (weights == nullptr) {
        throw std::invalid_argument("Qwen3::Model::Model: weights cannot be nullptr");
    }
    if (device_ids.empty()) {
        throw std::invalid_argument("Qwen3::Model::Model: device_ids cannot be empty");
    }

    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<DeviceResource<WeightsTensor>>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    // Launch device threads
    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice<WeightsTensor, Meta, Weights>, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i], Qwen3inferDeviceBatch);
    }
    // Wait for all devices to be ready
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

}; // namespace Qwen3
