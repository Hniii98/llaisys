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

    // 公共 score 缓冲
    auto score_storage =
    llaisys::core::context().runtime().allocateDeviceStorage(
        sizeof(float) * seq_len * nhead * total_len);

    float *score = reinterpret_cast<float*>(score_storage->memory());

    

    if (d == 128) {
        // 走专用 kernel
        dim3 block(128), grid(static_cast<unsigned>(seq_len), static_cast<unsigned>(nhead));
        naive_atten3d_hdim128_kernel<T>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<T*>(atten_val),
                reinterpret_cast<const T*>(q),
                reinterpret_cast<const T*>(k),
                reinterpret_cast<const T*>(v),
                score, scale, seq_len, nhead, total_len, nkvhead);
        
        
    } else {
        // 走通用 kernel
        constexpr int BLOCK_THREADS = 256;
        dim3 block(BLOCK_THREADS), grid(static_cast<unsigned>(seq_len), static_cast<unsigned>(nhead));

        size_t shmem_bytes = d * sizeof(float);

        atten3d_generic_kernel<T, BLOCK_THREADS>
            <<<grid, block, shmem_bytes, stream>>>(
                reinterpret_cast<T*>(atten_val),
                reinterpret_cast<const T*>(q),
                reinterpret_cast<const T*>(k),
                reinterpret_cast<const T*>(v),
                score, scale, seq_len, nhead, total_len, nkvhead, static_cast<int>(d));
    }

    // 4) 确保 GPU 端已完成再让 score_storage 析构释放显存
    CHECK_CUDA(cudaGetLastError());           
    CHECK_CUDA(cudaStreamSynchronize(stream));
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
