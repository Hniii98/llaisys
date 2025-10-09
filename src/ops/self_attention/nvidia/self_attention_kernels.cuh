#pragma once

#include "generic/atten3d_generic.cuh"
#include "hdim128/atten3d_hdim.cuh"

namespace llaisys::ops::nvidia::kernels {
static __global__ void cast_fp32_to_fp16(const float* __restrict__ src, __half* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}
static __global__ void cast_fp16_to_fp32(const __half* __restrict__ src, float* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}
static __global__ void cast_fp32_to_bf16(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16_rn(src[i]);
}
static __global__ void cast_bf16_to_fp32(const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}
} // namespace