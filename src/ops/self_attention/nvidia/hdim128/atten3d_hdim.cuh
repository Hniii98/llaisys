#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include "llaisys.h"

namespace llaisys::ops::nvidia::kernels {


// naive 128 prefill 专用核声明
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


template<typename T>
void atten3d_hdim128_decode_kernel(
	T *atten_val,  /* [seqlen, nhead, d=128] */
    const T *q,    /* [seqlen, nhead, d=128] */
    const T *k,    /* [total_len, nkvhead, d=128] */
    const T *v,    /* [total_len, nkvhead, d=128] */
	float scale,
	size_t seq_len,
	size_t nhead,
	size_t total_len,
	size_t nkvhead,
	cudaStream_t stream_in);

} // namespace llaisys::ops::nvidia::kernels


namespace llaisys::ops::nvidia::kernels {

// 通用 trait
template <typename T>
struct CudnnDType;

// half
template <>
struct CudnnDType<__half> {
    static inline constexpr cudnn_frontend::DataType_t fe_type = cudnn_frontend::DataType_t::HALF;
};

// bfloat16
template <>
struct CudnnDType<__nv_bfloat16> {
    static inline constexpr cudnn_frontend::DataType_t fe_type = cudnn_frontend::DataType_t::BFLOAT16;
};

}