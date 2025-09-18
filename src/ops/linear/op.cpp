#include "op.hpp"

#include "cpu/cpu_linear.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    /*
        Currently, only 2D input, output, and weight matrices are supported. 
        The weight matrix is assumed to be untransposed. 
        The bias term is optional, and no broadcasting is applied in the computation.
    */

    CHECK_SAME_SHAPE(in->shape()[1], weight->shape()[1]); // eg. [a, b] * [c, d], b must equal to d while 
                                                         // [c, d] is untransposed.
    CHECK_SAME_DEVICE(out, in, weight);                   
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() & weight->isContiguous(), 
        "Linear: all input should be contiguous.");   
                                                          
    if (bias) {
        CHECK_SAME_DEVICE(in, bias);
        CHECK_SAME_DTYPE(in->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "Linear: all input should be contiguous.");
    } 


    // always suport cpu caculation
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        if(bias) {
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), 
                               in->shape()[0], in->shape()[1], out->shape()[1], out->dtype()); 
        } else {
            return cpu::linear(out->data(), in->data(), weight->data(), nullptr, 
                               in->shape()[0], in->shape()[1], out->shape()[1], out->dtype());
        }
     
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());


    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:   
        if(bias) {
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), 
                               in->shape()[0], in->shape()[1], out->shape()[1], out->dtype()); 
        } else {
            return cpu::linear(out->data(), in->data(), weight->data(), nullptr, 
                               in->shape()[0], in->shape()[1], out->shape()[1], out->dtype());
        }   
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
