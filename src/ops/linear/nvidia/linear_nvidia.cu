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
void linear_(std::byte* out,
             const std::byte* in,
             const std::byte* weight,
             const std::byte* bias,
             size_t m,  // batch (rows of out)
             size_t k,  // in_dim
             size_t n,  // out_dim (cols of out)
             cudaStream_t stream) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 为了数值更贴近 PyTorch：float 用 DEFAULT（禁用 Tensor Core/TF32），
    // half/bf16 用 TENSOR_OP（走 Tensor Core）
    const auto algo =
        std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT
                                 : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    // Row-major 计算：out(m,n) = in(m,k) * weight^T(k,n)
    // 映射到 cuBLAS(列主序)的标准配方：
    // CUBLAS 计算：C(n,m) = op(A) * op(B)
    //   A = weight  (视作 k x n, lda = k),   opA = T -> (n x k)
    //   B = in      (视作 k x m, ldb = k),   opB = N -> (k x m)
    //   C = out     (视作 n x m, ldc = n)
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,                 // opA, opB
        static_cast<int>(n),                      // m' = n
        static_cast<int>(m),                      // n' = m
        static_cast<int>(k),                      // k
        &alpha,
        reinterpret_cast<const T*>(weight), CType, static_cast<int>(k), // A: lda = k
        reinterpret_cast<const T*>(in),     CType, static_cast<int>(k), // B: ldb = k
        &beta,
        reinterpret_cast<T*>(out),          CType, static_cast<int>(n), // C: ldc = n
        CUBLAS_COMPUTE_32F,
        algo));

    if (bias) {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>(m),
                  static_cast<unsigned>((n + block.x - 1u) / block.x));
        add_bias_kernel<T><<<grid, block, 0, stream>>>(
            reinterpret_cast<T*>(out),
            reinterpret_cast<const T*>(bias),
            static_cast<int>(m), static_cast<int>(n));
    }

    CHECK_CUBLAS(cublasDestroy(handle));
}


} // anomynous namespace


namespace llaisys::ops::nvidia {

void linear(std::byte* out, const std::byte* in, const std::byte* weight, const std::byte* bias,
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
