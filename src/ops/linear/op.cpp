#include "op.hpp"

#include "cpu/cpu_linear.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);
    
    ASSERT(out->isContiguous() && in->isContiguous() & weight->isContiguous(), 
           "Linear: all input should be contiguous.");

      // always suport cpu caculation
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::linear(out, in, weight, bias);  
        return;   
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());


    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:   
        return cpu::linear(out, in, weight, bias);    

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
