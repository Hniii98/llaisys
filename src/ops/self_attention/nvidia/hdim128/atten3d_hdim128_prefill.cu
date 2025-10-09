#include "../self_attention_kernels.cuh"
#include "../../../../device/nvidia/utils.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <thrust/functional.h>

#include "atten3d_hdim128.cuh"


namespace llaisys::ops::nvidia::kernels {

// naive self attention implementation for qwen2 1.5B of 128 hidden dimension
template<typename T>
__global__ void naive_atten3d_hdim128_kernel(
    T *atten_val,  /* [seqlen, nhead, d=128] */
    const T *q,    /* [seqlen, nhead, d=128] */
    const T *k,    /* [total_len, nkvhead, d=128] */
    const T *v,    /* [total_len, nkvhead, d=128] */
    float *score,  /* [seqlen, nhead, total_len] (writable: logits→weights) */
    float  scale,
    size_t seq_len,
    size_t nhead,
    size_t total_len,
    size_t nkvhead) {
        
    constexpr int HDIM = 128;

    const int    tid     = threadIdx.x;
    const size_t query_i = blockIdx.x;
    const size_t head_i  = blockIdx.y;

    using BlockReduce = cub::BlockReduce<float, HDIM>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    using llaisys::device::nvidia::utils::load_as_float;
    using llaisys::device::nvidia::utils::store_from_float;

    const size_t cache_len   = total_len - seq_len;
    const size_t kv_head_idx = head_i * nkvhead / nhead;
    const size_t max_k_idx   = query_i + cache_len;

    // load query row into shared memory
    __shared__ float query_i_shared[HDIM];
    if (tid < HDIM) {
        const size_t q_offset = (query_i * nhead + head_i) * (size_t)HDIM + tid;
        query_i_shared[tid] = load_as_float(q + q_offset);
    }
    __syncthreads();

    size_t score_offset = (query_i * nhead + head_i) * total_len;
    float *score_vec = score + score_offset;

    // 1. score = Q * K^T * scale  (+ causal mask)
    for (size_t key_i = 0; key_i < total_len; ++key_i) {
        const size_t k_offset = (key_i * nkvhead + kv_head_idx) * (size_t)HDIM + tid;
        const float  kd       = load_as_float(k + k_offset);
        const float  partial  = query_i_shared[tid] * kd;

        float dot = BlockReduce(temp_storage).Sum(partial);
        __syncthreads(); // reuse temp_storage

        if (tid == 0) {
            score_vec[key_i] = (key_i > max_k_idx) ? -CUDART_INF_F : (dot * scale);
        }
        __syncthreads();
    }

    // 2. causal softmax
    float local_max = -CUDART_INF_F;
    for (size_t i = tid; i <= max_k_idx; i += blockDim.x) {
        local_max = fmaxf(local_max, score_vec[i]);
    }
    float max_score = BlockReduce(temp_storage).Reduce(local_max, thrust::maximum<float>());
    

    // 广播 max_score
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

    // 广播 block_sum
    __shared__ float shared_block_sum;
    if (tid == 0) shared_block_sum = block_sum;
    __syncthreads();

    for (size_t i = tid; i < total_len; i += blockDim.x) {
        if (i <= max_k_idx) {
            score_vec[i] = score_vec[i] / shared_block_sum;
        } else {
            score_vec[i] = 0.0f;
        }
    }
    __syncthreads();

    //3. aggregate V: 每个线程负责一个通道
    float acc = 0.0f;
    const size_t v_channel_offset = kv_head_idx * HDIM + tid; // 每个线程负责一个通道
    for (size_t key_i = 0; key_i <= max_k_idx; ++key_i) {
        const float wi = score_vec[key_i];
        const size_t v_offset = (key_i * nkvhead) * HDIM + v_channel_offset;
        const float  v_elem = load_as_float(v + v_offset);
        acc += wi * v_elem;
    }

    // 写回结果
    const size_t out_offset = (query_i * nhead + head_i) * HDIM + tid;
    store_from_float(atten_val + out_offset, acc);
    }

// 显式实例化，host 端要能找到一个“入口符号”（内核函数指针），这个符号必须在编译时就生成。
template __global__ void naive_atten3d_hdim128_kernel<float>(
    float*, const float*, const float*, const float*, float*, float,
    size_t, size_t, size_t, size_t);
template __global__ void naive_atten3d_hdim128_kernel<__half>(
    __half*, const __half*, const __half*, const __half*, float*, float,
    size_t, size_t, size_t, size_t);
template __global__ void naive_atten3d_hdim128_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    float*, float, size_t, size_t, size_t, size_t);

} // namespace llaisys::ops::nvidia::kernels
