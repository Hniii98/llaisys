#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia::kernels {

// 通用核声明 (任意 d)
// atten_val: [seqlen, nhead, d]
// score:     [seqlen, nhead, total_len]
template<typename T, int BLOCK_THREADS = 256>
__global__ void atten3d_generic_kernel(
    T* atten_val,
    const T* q,
    const T* k,
    const T* v,
    float* score,
    float scale,
    size_t seq_len,
    size_t nhead,
    size_t total_len,
    size_t nkvhead,
    int d);

} // namespace llaisys::ops::nvidia::kernels