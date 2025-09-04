#ifndef _QWEN_MODEL_H_
#define _QWEN_MODEL_H_

#include "infinicore_infer/models/qwen3.h"

#include "qwen2moe/qwen2moe_model.hpp"
#include "qwen3/qwen3_model.hpp"
#include "qwen3moe/qwen3moe_model.hpp"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

template <typename Model, typename Meta, typename Weights>
Model *createModel(const Meta *meta,
                   const Weights *weights,
                   infiniDevice_t device,
                   int ndev,
                   const int *dev_ids) {
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    Model *model = new Model(meta, weights, device, device_ids);
    return model;
}

template <typename Model>
void destroyModel(Model *model) {
    auto ndev = model->dev_resources.size();

    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    delete model;
}

#endif