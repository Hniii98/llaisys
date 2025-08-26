#include "op.hpp"



#include "cpu/rearrange_cpu.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
   CHECK_SAME_DEVICE(in, out);
   CHECK_SAME_SHAPE(in->shape(), out->shape());
   CHECK_SAME_DTYPE(in->dtype(), out->dtype());
   ASSERT(out->isContiguous(), "Rearrange: output tensor shoulbe be contiguous.");
   
    // always suport cpu caculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(), in->shape(), in->strides(),
                              in->dtype());
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());


    switch (out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::rearrange(out->data(), in->data(), in->shape(),  in->strides(),
                                  in->dtype());
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
