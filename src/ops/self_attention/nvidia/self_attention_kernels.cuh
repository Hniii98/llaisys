#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace llaisys::ops::nvidia::kernels {

// naive 128 专用核声明
// atten_val: [seqlen, nhead, 128]
// q, k, v:   see layout in implementation
// score:     [seqlen, nhead, total_len]
template<typename T>
__global__ void naive_atten3d_hdim128_kernel(
    T* atten_val,
    const T* q,
    const T* k,
    const T* v,
    float* score,
    float scale,
    size_t seq_len,
    size_t nhead,
    size_t total_len,
    size_t nkvhead);

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
