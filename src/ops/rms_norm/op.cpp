#include "op.hpp"

#include "../../core/llaisys_core.hpp"

#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    // Only support contiguous inputs with same shape for now.
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(out->isContiguous() && weight->isContiguous() && in->isContiguous(), 
            "RMSNorm: all input should be contiguous.");

    size_t N = in->shape()[0], D = in->shape()[1];

    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(),
                             N, D, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(),
                                N, D, eps);
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
