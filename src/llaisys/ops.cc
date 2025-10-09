#include "llaisys/ops.h"

#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

__C {
    void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
        llaisys::ops::add(c->tensor, a->tensor, b->tensor);
    }
    void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
        llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor);
    }
    void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
        llaisys::ops::embedding(out->tensor, index->tensor, weight->tensor);
    }
    void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
        llaisys::ops::linear(out->tensor, in->tensor, weight->tensor, bias->tensor);
    }
    void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
        llaisys::ops::rearrange(out->tensor, in->tensor);
    }
    void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
        llaisys::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps);
    }
    void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
        llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta);
    }

    void llaisysSelfAttention(llaisysTensor_t attn_val,
                          llaisysTensor_t q,
                          llaisysTensor_t k,
                          llaisysTensor_t v,
                          float scale) noexcept {
    try {
        llaisys::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
    } catch (const std::exception& e) {
        // 捕获所有从std::exception派生的异常
        fprintf(stderr,
                "[ERROR] llaisysSelfAttention failed: %s\n"
                "  attn_val=%p  q=%p  k=%p  v=%p  scale=%.3f\n",
                e.what(), (void*)attn_val, (void*)q, (void*)k, (void*)v, scale);
        fflush(stderr);
        return;
    } catch (...) {
        // 捕获所有其他类型的异常
        fprintf(stderr,
                "[ERROR] llaisysSelfAttention: unknown exception\n"
                "  attn_val=%p  q=%p  k=%p  v=%p  scale=%.3f\n",
                (void*)attn_val, (void*)q, (void*)k, (void*)v, scale);
        fflush(stderr);
        return;
    }
    }


    void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
        llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor);
    }
}
