#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "llaisys.h"

namespace llaisys::ops::nvidia::kernels {

template<typename T> struct CudaDataTypeOf;
template<> struct CudaDataTypeOf<float>        { static constexpr cudaDataType_t value = CUDA_R_32F; };
template<> struct CudaDataTypeOf<__half>       { static constexpr cudaDataType_t value = CUDA_R_16F; };
template<> struct CudaDataTypeOf<__nv_bfloat16>{ static constexpr cudaDataType_t value = CUDA_R_16BF; };



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

// 128 score 计算 声明
// q, k, v:   [total_len, nkvhead, 128]
// score:     [seqlen, nhead, total_len]
template<typename T>
__global__ void atten3d_hdim128_score_kernel(
    const T *q,    /* [seqlen, nhead, d=128] */
    const T *k,    /* [total_len, nkvhead, d=128] */
    const T *v,    /* [total_len, nkvhead, d=128] */
    float *score,  /* [seqlen, nhead, total_len] (writable: logits→weights) */
    float  scale,
    size_t seq_len,
    size_t nhead,
    size_t total_len,
    size_t nkvhead);

// 128 score 计算 声明
// atten_val: [seqlen, nhead, 128]
// v		: [total_len, nkvhead, 128]
// score:     [seqlen, nhead, total_len]
template<typename T, cudaDataType_t CType>
void atten3d_hdim128_vproj_kernel(
    T*       atten_val,   // [seqlen, nhead, d=128]  row-major
    const T* v,           // [total_len, nkvhead, d=128] row-major
    const float* score,   // [seqlen, nhead, total_len]  row-major
    size_t   seq_len,
    size_t   nhead,
    size_t   total_len,
    size_t   nkvhead,
    cudaStream_t stream_in);




template<typename T>
inline void launch_atten3d_hdim128_vproj(
    T*       atten_val,
    const T* v,
    const float* score,
    size_t   seq_len,
    size_t   nhead,
    size_t   total_len,
    size_t   nkvhead,
    cudaStream_t stream) {
    // 防止 CType 映射没生效被实例成 <T, 0>
    static_assert(CudaDataTypeOf<T>::value == CUDA_R_32F ||
                  CudaDataTypeOf<T>::value == CUDA_R_16F ||
                  CudaDataTypeOf<T>::value == CUDA_R_16BF, "Unsupported T for Lt");

    atten3d_hdim128_vproj_kernel<T, CudaDataTypeOf<T>::value>(
        atten_val, v, score, seq_len, nhead, total_len, nkvhead, stream);
}

} // namespace llaisys::ops::nvidia::kernels