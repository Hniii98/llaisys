#include "self_attention_kernels.cuh"
#include "../../../device/nvidia/utils.cuh"
#include <cub/cub.cuh>
#include <math_constants.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


#include <thrust/functional.h>

namespace llaisys::ops::nvidia::kernels {

// generic self attention implementation for arbitrary hidden dimension d
template<typename T, int BLOCK_THREADS>
__global__ void atten3d_generic_kernel(
    T *atten_val,  /* [seqlen, nhead, d] */
    const T *q,    /* [seqlen, nhead, d] */
    const T *k,    /* [total_len, nkvhead, d] */
    const T *v,    /* [total_len, nkvhead, d] */
    float *score,  /* [seqlen, nhead, total_len] (logits→weights) */
    float scale,
    size_t seq_len,
    size_t nhead,
    size_t total_len,
    size_t nkvhead,
    int d) {

    const int    tid     = threadIdx.x;
    const size_t query_i = blockIdx.x;
    const size_t head_i  = blockIdx.y;

    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    using llaisys::device::nvidia::utils::load_as_float;
    using llaisys::device::nvidia::utils::store_from_float;

    const size_t cache_len   = total_len - seq_len;
    const size_t kv_head_idx = head_i * nkvhead / nhead;
    const size_t max_k_idx   = query_i + cache_len; 
    
    // 共享内存存放当前 query 行
    extern __shared__ float query_shared[];
    for (int t = tid; t < d; t += blockDim.x) {
        const size_t q_offsetset = (query_i * nhead + head_i) * (size_t)d + t;
        query_shared[t] = load_as_float(q + q_offsetset);
    }
    __syncthreads();

    float *score_vec = score + (query_i * nhead + head_i) * total_len;

    // 1) score = Q * K^T * scale （带 causal mask）
    for (size_t key_i = 0; key_i < total_len; key_i++) {
        float partial = 0.0f;
       
        for (int t = tid; t < d; t += blockDim.x) {
            const size_t k_offset = (key_i * nkvhead + kv_head_idx) * (size_t)d + t;
            const float kd = load_as_float(k + k_offset);
            partial = fmaf(query_shared[t], kd, partial);
        }
        float dot = BlockReduce(temp_storage).Sum(partial);
        __syncthreads();

        if (tid == 0) {
            score_vec[key_i] = (key_i > max_k_idx) ? -CUDART_INF_F : (dot * scale);
        }
        __syncthreads();
    }

    // 2) causal softmax（数值稳定）
    float local_max = -CUDART_INF_F;
    for (size_t i = tid; i <= max_k_idx; i += blockDim.x) {
        local_max = fmaxf(local_max, score_vec[i]);
    }
    
    float max_score = BlockReduce(temp_storage).Reduce(local_max, thrust::maximum<float>());
    __syncthreads();

    __shared__ float shared_max_score;
    if (tid == 0) shared_max_score = max_score;
    __syncthreads();

    float local_sum = 0.0f;
    for (size_t i = tid; i <= max_k_idx; i += blockDim.x) {
        float ei = expf(score_vec[i] - shared_max_score);
        score_vec[i] = ei;
        local_sum += ei;
    }
    float block_sum = BlockReduce(temp_storage).Sum(local_sum);
    __syncthreads();

    __shared__ float shared_block_sum;
    if (tid == 0) shared_block_sum = block_sum;
    __syncthreads();

    for (size_t i = tid; i < total_len; i += blockDim.x) {
        score_vec[i] = (i <= max_k_idx) ? (score_vec[i] / shared_block_sum) : 0.0f;
    }
    __syncthreads();

    // 3) 聚合 V
    for (int t = tid; t < d; t += blockDim.x) {
        float acc = 0.0f;
        for (size_t key_i = 0; key_i <= max_k_idx; key_i++) {
            const float wi = score_vec[key_i];
            const size_t v_offset = (key_i * nkvhead + kv_head_idx) * (size_t)d + t;
            const float v_elem = load_as_float(v + v_offset);
            acc = fmaf(wi, v_elem, acc);
        }
        const size_t out_offset = (query_i * nhead + head_i) * (size_t)d + t;
        store_from_float(atten_val + out_offset, acc);
    }
}

// 显式实例化
template __global__ void atten3d_generic_kernel<float, 256>(
    float*, const float*, const float*, const float*, float*, float,
    size_t, size_t, size_t, size_t, int);
template __global__ void atten3d_generic_kernel<__half, 256>(
    __half*, const __half*, const __half*, const __half*, float*, float,
    size_t, size_t, size_t, size_t, int);
template __global__ void atten3d_generic_kernel<__nv_bfloat16, 256>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    float*, float, size_t, size_t, size_t, size_t, int);

} // namespace llaisys::ops::nvidia::kernels
