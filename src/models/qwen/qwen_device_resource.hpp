#ifndef _QWEN_DEVICE_RESOURCE_
#define _QWEN_DEVICE_RESOURCE_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "../jiuge/jiuge_impl.hpp"

using DeviceResource = JiugeDeviceResource;

template <typename WeightsTensor, typename Meta, typename Weights>
void createDeviceResource(DeviceResource *rsrc, const Meta *meta,
                          const Weights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    if (!memory_pool) {
        printf("  ===> memory_pool failed \n !!");
    }

    if (meta) {
        // meta->print_info();
    }

    if (weights) {
        // weights->print_info();
    }

    const void *weights_tensor_ptr = new WeightsTensor(meta, weights, idev, ndev);
    if (weights_tensor_ptr) {
        // const WeightsTensor *ptr = reinterpret_cast<const WeightsTensor *>(weights_tensor_ptr);
        // ptr->print_info();
    }

    *rsrc = DeviceResource{
        device,
        dev_id,
        handle,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        stream,
        comm,
        memory_pool,
        weights_tensor_ptr};
    RUN_INFINI(infinirtDeviceSynchronize());
}

template <typename WeightsTensor>
void releaseDeviceResource(DeviceResource &res) {

    infinirtDeviceSynchronize();

    // Release individual Tensors
    if (res.weights_tensor_ptr) {
        const WeightsTensor *ptr = static_cast<const WeightsTensor *>(res.weights_tensor_ptr);
        delete ptr;
        res.weights_tensor_ptr = nullptr;
    }

    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

//
//
//
template <typename WeightsTensor, typename Meta, typename Weights>
void launchDevice(const Meta &meta, const Weights *weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm,
                  void (*inferDeviceBatch)(const Meta *, DeviceResource &, uint32_t, uint32_t, const uint32_t *, uint32_t, const uint32_t *, uint32_t, const uint32_t *, struct KVCache **kv_caches, const float *, const uint32_t *, const float *, uint32_t *, void *)) {

    // Create Device Resource
    createDeviceResource<WeightsTensor, Meta, Weights>(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);
    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    // Infer Loop
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;
        }

        inferDeviceBatch(&meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                         req.req_lens, req.nreq, req.req_pos, req.kv_caches,
                         req.temperature, req.topk, req.topp, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource<WeightsTensor>(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}

template <typename Model>
void inferBatch(Model *model,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                KVCache **kv_caches,
                const float *temperature, const uint32_t *topk, const float *topp,
                uint32_t *output) {

    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

template <typename Model>
void forwardBatch(Model *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  KVCache **kv_caches,
                  void *logits) {

    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = nullptr;
    model->req.logits = logits;
    model->req.temperature = nullptr;
    model->req.topk = nullptr;
    model->req.topp = nullptr;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

#endif