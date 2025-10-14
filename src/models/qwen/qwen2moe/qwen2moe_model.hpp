#ifndef _QWEN2MOE_MODEL_HPP_
#define _QWEN2MOE_MODEL_HPP_
#include "qwen2moe_weight.hpp"

namespace Qwen2MoE {

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
}; // namespace Qwen2MoE
#endif