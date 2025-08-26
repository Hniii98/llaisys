#include "op.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(q, k, v, attn_val);
    // Only support contiguous inputs with same shape for now. 
    CHECK_SAME_DTYPE(q->dtype(), k->dtype(), v->dtype());
    ASSERT(q->isContiguous() && k->isContiguous() && v->isContiguous() && attn_val, 
           "Self_Attention: all tensors must be contiguous.");
    ASSERT((q->ndim() == k->ndim()) && (k->ndim() == v->ndim()) && (q->ndim() == 3), "Self_Attention: Q K V should own 3 dimensions.");
    
    /*
    attn_val：[seqlen, nhead, dv]
           q：[seqlen, nhead, d]
           k：[total_len, nkvhead, d]
           v：[total_len, nkvhead, dv]
       scale：1 / sqrt(d)

    nkvhead may isn't equal to n head when doing GQA
    */
    

    size_t seq_len = q->shape()[0];
    size_t nhead = q->shape()[1]; 
    size_t d = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];
    
    ASSERT(nhead % nkvhead == 0, "Self_attention: nhead must be divisible by nkvhead.");
    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                    seq_len, nhead, d, total_len, nkvhead, dv);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                    seq_len, nhead, d, total_len, nkvhead, dv);
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
