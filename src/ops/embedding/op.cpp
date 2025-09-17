#include "op.hpp"
#include <cstddef>

#include "cpu/embedding_cpu.hpp"
#include "nvidia/embedding_nvidia.cuh"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index should be int64.");
    // only support contiguous weight for now.
    ASSERT(weight->isContiguous(), "Embedding: weight shoule be contiguouts.");
    ASSERT(weight->ndim() == 2 && index->ndim() == 1, "Embedding: supoort 2D input and 1D index for now.");
    ASSERT(index->shape()[0] <= weight->shape()[0], "Embedding: index should in weight shape boundary!");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), index->shape()[0], 
                              weight->data(), weight->strides()[0], weight->dtype());
    }
    
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), index->shape()[0], 
                              weight->data(), weight->strides()[0], weight->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), index->shape()[0],
                weight->data(), weight->strides()[0], weight->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}
} // namespace llaisys::ops
