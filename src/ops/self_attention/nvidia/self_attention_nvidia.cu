#include "self_attention_nvidia.cuh"
#include "self_attention_kernels.cuh"
#include "../../../device/nvidia/utils.cuh"
#include "../../../core/llaisys_core.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

using namespace llaisys::ops::nvidia::kernels;



template<typename T>
void selfatten3d_dispatch(std::byte *atten_val, const std::byte *q, const std::byte *k, const std::byte *v,
                          float scale, size_t seq_len, size_t nhead, size_t d,
                          size_t total_len, size_t nkvhead, size_t dv) {

    ASSERT(dv == d, "self attention kernel: d should be equal to dv.");
    auto stream = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    // Decode fast path 
    if (seq_len == 1) {
        atten3d_hdim128_decode_kernel<T>(
            reinterpret_cast<T*>(atten_val),
            reinterpret_cast<const T*>(q),
            reinterpret_cast<const T*>(k),
            reinterpret_cast<const T*>(v),
            scale, seq_len, nhead, total_len, nkvhead, stream);
    }
    // Prefill 
    else {
        // 公共 score 缓冲
        auto score_storage =
        llaisys::core::context().runtime().allocateDeviceStorage(
            sizeof(float) * seq_len * nhead * total_len);
        float *score = reinterpret_cast<float*>(score_storage->memory());
        if (d == 128) {
            // 专用 naive kernel
            dim3 block(128);
            dim3 grid(static_cast<unsigned>(seq_len),
                      static_cast<unsigned>(nhead));
            naive_atten3d_hdim128_kernel<T>
                <<<grid, block, 0, stream>>>(
                    reinterpret_cast<T*>(atten_val),
                    reinterpret_cast<const T*>(q),
                    reinterpret_cast<const T*>(k),
                    reinterpret_cast<const T*>(v),
                    score, scale, seq_len, nhead, total_len, nkvhead);
        } else {
            // 通用 kernel
            constexpr int BLOCK_THREADS = 256;
            dim3 block(BLOCK_THREADS);
            dim3 grid(static_cast<unsigned>(seq_len),
                      static_cast<unsigned>(nhead));
            size_t shmem_bytes = d * sizeof(float);

            atten3d_generic_kernel<T, BLOCK_THREADS>
                <<<grid, block, shmem_bytes, stream>>>(
                    reinterpret_cast<T*>(atten_val),
                    reinterpret_cast<const T*>(q),
                    reinterpret_cast<const T*>(k),
                    reinterpret_cast<const T*>(v),
                    score, scale, seq_len, nhead, total_len,
                    nkvhead, static_cast<int>(d));
        }
    }

}

// 对 float 做特化
template<>
void selfatten3d_dispatch<float>(std::byte *atten_val, const std::byte *q, const std::byte *k, const std::byte *v,
                                 float scale, size_t seq_len, size_t nhead, size_t d,
                                 size_t total_len, size_t nkvhead, size_t dv) {
    ASSERT(dv == d, "self attention kernel: d should be equal to dv.");
    auto stream = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    if (seq_len > 1) {
        // prefill 
        auto score_storage = llaisys::core::context().runtime()
            .allocateDeviceStorage(sizeof(float) * seq_len * nhead * total_len);
        float *score = reinterpret_cast<float*>(score_storage->memory());

        if (d == 128) {
            dim3 block(128);
            dim3 grid((unsigned)seq_len, (unsigned)nhead);
            naive_atten3d_hdim128_kernel<float><<<grid, block, 0, stream>>>(
                reinterpret_cast<float*>(atten_val),
                reinterpret_cast<const float*>(q),
                reinterpret_cast<const float*>(k),
                reinterpret_cast<const float*>(v),
                score, scale, seq_len, nhead, total_len, nkvhead);
        } else {
            constexpr int BLOCK_THREADS = 256;
            dim3 block(BLOCK_THREADS);
            dim3 grid((unsigned)seq_len, (unsigned)nhead);
            size_t shmem_bytes = d * sizeof(float);
            atten3d_generic_kernel<float, BLOCK_THREADS><<<grid, block, shmem_bytes, stream>>>(
                reinterpret_cast<float*>(atten_val),
                reinterpret_cast<const float*>(q),
                reinterpret_cast<const float*>(k),
                reinterpret_cast<const float*>(v),
                score, scale, seq_len, nhead, total_len, nkvhead, (int)d);
        }
        return;
    }

    // ===== decode：FP32 → (BF16/FP16) → SDPA(FE, compute=FP32) → (可选)回写 FP32 =====
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    // bool use_bf16 = (prop.major >= 8); // Ampere+ 走 BF16，更稳健
    bool use_bf16 = false;
    const int64_t S_q=seq_len, H_q=nhead, D=d;
    const int64_t S_kv=total_len, H_kv=nkvhead;
    size_t nQ = size_t(S_q)  * H_q * D;
    size_t nK = size_t(S_kv) * H_kv * D;
    size_t nV = nK, nO = nQ;

    dim3 blk(256);
    dim3 grdQ((static_cast<unsigned>(nQ) + blk.x - 1) / blk.x);
    dim3 grdKV((static_cast<unsigned>(nK) + blk.x - 1) / blk.x);

    if (use_bf16) {
        auto qB = llaisys::core::context().runtime().allocateDeviceStorage(nQ * sizeof(__nv_bfloat16));
        auto kB = llaisys::core::context().runtime().allocateDeviceStorage(nK * sizeof(__nv_bfloat16));
        auto vB = llaisys::core::context().runtime().allocateDeviceStorage(nV * sizeof(__nv_bfloat16));
        auto oB = llaisys::core::context().runtime().allocateDeviceStorage(nO * sizeof(__nv_bfloat16));
        auto Qb = reinterpret_cast<__nv_bfloat16*>(qB->memory());
        auto Kb = reinterpret_cast<__nv_bfloat16*>(kB->memory());
        auto Vb = reinterpret_cast<__nv_bfloat16*>(vB->memory());
        auto Ob = reinterpret_cast<__nv_bfloat16*>(oB->memory());

        cast_fp32_to_bf16<<<grdQ, blk, 0, stream>>>(reinterpret_cast<const float*>(q), Qb, nQ);
        cast_fp32_to_bf16<<<grdKV, blk, 0, stream>>>(reinterpret_cast<const float*>(k), Kb, nK);
        cast_fp32_to_bf16<<<grdKV, blk, 0, stream>>>(reinterpret_cast<const float*>(v), Vb, nV);

        // 这里调用你修正后的 SDPA kernel（K/V 的 H 维 = H_kv；generate_stats=false；compute=FP32）
        atten3d_hdim128_decode_kernel<__nv_bfloat16>(
            Ob, Qb, Kb, Vb, scale, seq_len, nhead, total_len, nkvhead, stream);

        // 如果上层需要 FP32 输出，转回（你的 atten_val 就是 float*）
        cast_bf16_to_fp32<<<grdQ, blk, 0, stream>>>(Ob, reinterpret_cast<float*>(atten_val), nO);
    } else {
        auto qH = llaisys::core::context().runtime().allocateDeviceStorage(nQ * sizeof(__half));
        auto kH = llaisys::core::context().runtime().allocateDeviceStorage(nK * sizeof(__half));
        auto vH = llaisys::core::context().runtime().allocateDeviceStorage(nV * sizeof(__half));
        auto oH = llaisys::core::context().runtime().allocateDeviceStorage(nO * sizeof(__half));
        auto Qh = reinterpret_cast<__half*>(qH->memory());
        auto Kh = reinterpret_cast<__half*>(kH->memory());
        auto Vh = reinterpret_cast<__half*>(vH->memory());
        auto Oh = reinterpret_cast<__half*>(oH->memory());

        cast_fp32_to_fp16<<<grdQ, blk, 0, stream>>>(reinterpret_cast<const float*>(q), Qh, nQ);
        cast_fp32_to_fp16<<<grdKV, blk, 0, stream>>>(reinterpret_cast<const float*>(k), Kh, nK);
        cast_fp32_to_fp16<<<grdKV, blk, 0, stream>>>(reinterpret_cast<const float*>(v), Vh, nV);

        atten3d_hdim128_decode_kernel<__half>(
            Oh, Qh, Kh, Vh, scale, seq_len, nhead, total_len, nkvhead, stream);

        cast_fp16_to_fp32<<<grdQ, blk, 0, stream>>>(Oh, reinterpret_cast<float*>(atten_val), nO);
    }

}


} // anonymous namespace


namespace llaisys::ops::nvidia {

void self_attention(std::byte *atten_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t type, size_t seq_len, size_t nhead, size_t d,
                    size_t total_len, size_t nkvhead, size_t dv) {

    switch (type) {
        case LLAISYS_DTYPE_F32:
            selfatten3d_dispatch<float>(atten_val, q, k, v, scale, seq_len, nhead, d,
                                        total_len, nkvhead, dv);
            break;
        case LLAISYS_DTYPE_F16:
            selfatten3d_dispatch<__half>(atten_val, q, k, v, scale, seq_len, nhead, d,
                                         total_len, nkvhead, dv);
            break;
        case LLAISYS_DTYPE_BF16:
            selfatten3d_dispatch<__nv_bfloat16>(atten_val, q, k, v, scale, seq_len, nhead, d,
                                                total_len, nkvhead, dv);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
