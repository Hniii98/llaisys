#include "op.hpp"

#include "../../core/llaisys_core.hpp"

#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_SHAPE(in->shape(), out->shape());
    CHECK_SAME_DTYPE(in->dtype(), out->dtype());
    ASSERT(in->ndim() == 3, "RoPe: input should be 3 dimension.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), 
           "RoPe: all input should be contiguous.");
    size_t seqlen = in->shape()[0], nhead = in->shape()[1], d = in->shape()[2];
    
    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
                  seqlen, nhead, d);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
                  seqlen, nhead, d);
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
