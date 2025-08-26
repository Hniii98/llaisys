#include "op.hpp"
#include <cstddef>

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index should be int64.");
    // only support contiguous weight for now.
    ASSERT(weight->isContiguous(), "Embedding: weight shoule be contiguouts.");
    ASSERT(weight->ndim() == 2 && index->ndim() == 1, "Embedding: supoort 2D input and 1D index for now.");
    ASSERT(index->shape()[0] <= weight->shape()[0], "Embedding: index should in weight shape boundary!");

    int64_t* row_list = reinterpret_cast<int64_t*>(index->data());
    
    // only support 1D index and 2D weight.
    size_t embedding_dim = weight->shape()[1];
    size_t numrow = index->shape()[0];
    ptrdiff_t stride = weight->strides()[0];

    // always suport cpu caculation
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        for (size_t i = 0; i < numrow; i++) { // iterate row_list
            size_t line_idx = row_list[i];  // get a line number
            size_t src_offset = line_idx * stride * llaisys::utils::dsize(weight->dtype()); // units: byte
            size_t dst_offset = i * stride * llaisys::utils::dsize(weight->dtype());; // units: byte
            cpu::embedding(out->data() + dst_offset, weight->data() + src_offset, weight->dtype(),
                             embedding_dim);   
        }
        return;
        
    }
    
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());


    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        for (size_t i = 0; i < numrow; i++) {
            size_t line_idx = row_list[i];  // get line number
            size_t src_offset = line_idx * stride * llaisys::utils::dsize(weight->dtype()); // units: byte
            size_t dst_offset = i * stride; // units: byte
            cpu::embedding(out->data() + dst_offset, weight->data() + src_offset, weight->dtype(),
                             embedding_dim);
        }
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}
} // namespace llaisys::ops
