#ifndef _QWEN3_MODEL_HPP_
#define _QWEN3_MODEL_HPP_
#include "qwen3_weight.hpp"

namespace Qwen3 {

struct Model {
    Meta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<DeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    Model(const Meta *, const Weights *, infiniDevice_t device, std::vector<int> device_ids);
};
}; // namespace Qwen3
#endif