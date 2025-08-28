#pragma once
#include "tensor/tensor.hpp"
#include "llaisys/llaisys_tensor.hpp"

namespace llaisys {

// 把 C++ shared_ptr<Tensor> 转成 C 句柄（裸指针，不转移所有权）
inline llaisysTensor_t to_c_handle(const tensor_t& t) {
    return t ? reinterpret_cast<llaisysTensor_t>(t.get()) : nullptr;
}

// 从 C 句柄“借用”为 shared_ptr（不拥有、不会销毁底层 Tensor）
// 注意：这个 shared_ptr 不会释放底层 Tensor，
// 真实的内存释放仍然由 tensorDestroy(...) 负责。
// 用途：方便算子/实现里以 tensor_t (shared_ptr<Tensor>) 形式操作。
inline tensor_t borrow(llaisysTensor_t h) {
    if (!h) return nullptr;
    return tensor_t(reinterpret_cast<Tensor*>(h), [](Tensor*){/* no-op */});
}

} // namespace llaisys
