#include "llaisys.h"
#include "../../ops.hpp"
#include "../../../device/nvidia/utils.cuh"


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


namespace {


template <typename T>
__global__ void add_bias_kernel(T* out, const T* bias, int m, int n) {
    int row = blockIdx.x;
    int col = threadIdx.x + blockDim.x * blockIdx.y;
    if (row < m && col < n) {
        out[row * n + col] += bias[col];
    }
}


template <typename T, cudaDataType_t CType>
void linear_(std::byte *out, // [m, n]  (row major)  /  [n, m] (col major)
             const std::byte *in, // [m, k] (row major) / [k, m] (col major)
             const std::byte *weight, // [n, k] (row major) / [k, n] (col major)
             const std::byte *bias,
             size_t m,  // batch (rows of out)
             size_t k,  // in_dim
             size_t n,  // out_dim (cols of out)
             cudaStream_t stream) {

    auto &runtime = llaisys::core::context().runtime();

    auto handle = runtime.cublasHandle();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 为了数值更贴近 PyTorch：float 用 DEFAULT（禁用 Tensor Core/TF32），
    // half/bf16 用 TENSOR_OP（走 Tensor Core）
    const auto algo =
        std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT
                                 : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    // 数学目标（row 语义，不涉及存储）：
    //   out(m,n) = in(m,k) * weight^T(k,n)
    //
    // 关键等价（不拷贝、不物理转置）：
    //   (in * weight^T)^T = weight * in^T
    //
    // 存储等价（字节层面）：
    //   row(m,n) <=> col(n,m)
    //
    // 于是我们在 cuBLAS（列主序）里让它计算：
    //   C(n,m) = weight(n,k) * in^T(k,m) = out^T(n,m)
    //
    // 指针与参数映射（只改变“解释方式”，不做物理转置）：
    //   A <- weight  // row: [n,k]  → cuBLAS(col视角): [k,n],  lda = k,  opA = T  → (n,k)
    //   B <- in      // row: [m,k]  → cuBLAS(col视角): [k,m],  ldb = k,  opB = N  → (k,m) = in^T
    //   C <- out     // row: [m,n]  → cuBLAS(col视角): [n,m],  ldc = n
    //
    //   维度：m' = n,  n' = m,  k' = k
    //
    // opA/opB 是“数学转置标志”，用于纠正 cuBLAS 的列主序解释；不进行任何物理数据转置/拷贝。
    auto compute_type =
        std::is_same_v<T, __half>      ? CUBLAS_COMPUTE_32F_FAST_16F :
        std::is_same_v<T, __nv_bfloat16> ? CUBLAS_COMPUTE_32F_FAST_16BF :
        CUBLAS_COMPUTE_32F;



    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,                 // opA, opB
        static_cast<int>(n),                      // m' = n
        static_cast<int>(m),                      // n' = m
        static_cast<int>(k),                      // k
        &alpha,
        reinterpret_cast<const T *>(weight), CType, static_cast<int>(k), // A: lda = k
        reinterpret_cast<const T *>(in),     CType, static_cast<int>(k), // B: ldb = k
        &beta,
        reinterpret_cast<T *>(out),          CType, static_cast<int>(n), // C: ldc = n
        compute_type,
        algo));

    if (bias) {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>(m),
                  static_cast<unsigned>((n + block.x - 1u) / block.x));
        add_bias_kernel<T><<<grid, block, 0, stream>>>(
            reinterpret_cast<T *>(out),
            reinterpret_cast<const T *>(bias),
            static_cast<int>(m), static_cast<int>(n));
    }

}


} // anomynous namespace


namespace llaisys::ops::nvidia {

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            size_t sequence_length, size_t embedding_dim, size_t features_dim,
            llaisysDataType_t type) {

    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    switch (type) {
		case LLAISYS_DTYPE_F32:
			linear_<float, CUDA_R_32F>(out, in, weight, bias,
									   sequence_length, embedding_dim, features_dim, s);
			break;
		case LLAISYS_DTYPE_F16:
			linear_<__half, CUDA_R_16F>(out, in, weight, bias,
										sequence_length, embedding_dim, features_dim, s);
			break;
		case LLAISYS_DTYPE_BF16:
			linear_<__nv_bfloat16, CUDA_R_16BF>(out, in, weight, bias,
												sequence_length, embedding_dim, features_dim, s);
			break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
