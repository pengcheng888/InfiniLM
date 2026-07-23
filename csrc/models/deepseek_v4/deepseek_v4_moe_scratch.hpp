#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4FlatScratchBuffer {
public:
    infinicore::Tensor get(const infinicore::Shape &shape,
                           infinicore::DataType dtype,
                           const infinicore::Device &device) {
        const size_t required = numel(shape);
        if (!buffer_ || capacity_ < required || buffer_->dtype() != dtype || buffer_->device() != device) {
            buffer_ = infinicore::Tensor::empty({required}, dtype, device);
            capacity_ = required;
        }
        if (buffer_->numel() == required) {
            return buffer_->view(shape);
        }
        return buffer_->narrow({{0, 0, required}})->view(shape);
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    static size_t numel(const infinicore::Shape &shape) {
        size_t result = 1;
        for (const auto dim : shape) {
            result *= dim;
        }
        return result;
    }

    infinicore::Tensor buffer_;
    size_t capacity_{0};
};

class DeepseekV4SharedExpertScratch {
public:
    infinicore::Tensor activated(size_t tokens,
                                 size_t intermediate_size,
                                 infinicore::DataType dtype,
                                 const infinicore::Device &device) {
        return activated_.get({tokens, intermediate_size}, dtype, device);
    }

private:
    DeepseekV4FlatScratchBuffer activated_;
};

class DeepseekV4RoutedExpertScratch {
public:
    infinicore::Tensor output(size_t tokens,
                              size_t hidden_size,
                              infinicore::DataType dtype,
                              const infinicore::Device &device) {
        return output_.get({tokens, hidden_size}, dtype, device);
    }

    infinicore::Tensor contiguous_hidden(size_t tokens,
                                         size_t hidden_size,
                                         infinicore::DataType dtype,
                                         const infinicore::Device &device) {
        return contiguous_hidden_.get({tokens, hidden_size}, dtype, device);
    }

private:
    DeepseekV4FlatScratchBuffer output_;
    DeepseekV4FlatScratchBuffer contiguous_hidden_;
};

} // namespace infinilm::models::deepseek_v4
