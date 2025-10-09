#pragma once
#include "tensor/tensor.hpp"
#include "llaisys/llaisys_tensor.hpp"

namespace llaisys {

// 返回一个新的 C 句柄（包装对象）；调用方需 tensorDestroy() 释放该句柄。
inline llaisysTensor_t to_c_handle(const tensor_t& t) {
    if (!t) return nullptr;
    auto* wrap = new LlaisysTensor{};
    wrap->tensor = t;  // 引用计数+1
    return reinterpret_cast<llaisysTensor_t>(wrap);
}

inline tensor_t borrow(llaisysTensor_t h) {
    if (!h) return nullptr;
    auto* wrap = reinterpret_cast<LlaisysTensor*>(h); // 将不透明指针转换为真实定义才能取出智能指针
    return wrap->tensor;  // 不会释放底层 Tensor
}



} // namespace llaisys
