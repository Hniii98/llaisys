#include "llaisys.h"

#include "../self_attention_kernels.cuh"
#include "../../../../device/nvidia/utils.cuh"
#include "../../../../core/llaisys_core.hpp"



#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#include <vector>


namespace llaisys::ops::nvidia::kernels {

template<typename T, cudaDataType_t CType>
void atten3d_hdim128_vproj_kernel(
    T*       atten_val,   // [seqlen, nhead, d=128]  row-major
    const T* v,           // [total_len, nkvhead, d=128] row-major
    const float* score,   // [seqlen, nhead, total_len]  row-major
    size_t   seq_len,
    size_t   nhead,
    size_t   total_len,
    size_t   nkvhead,
    cudaStream_t stream_in) {

    constexpr int HDIM = 128;

	ASSERT(seq_len && nhead && total_len && nkvhead, "bad dims");
    ASSERT((nhead % nkvhead) == 0, "GQA requires nhead % nkvhead == 0");
    ASSERT(atten_val && v && score, "null device ptr");

    // cuBLASLt 句柄
    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

	

    // Matmul 描述符
    cublasLtMatmulDesc_t opDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    {
        cublasOperation_t transa = CUBLAS_OP_N; // A = score: [M=seqlen, K=total_len]
        cublasOperation_t transb = CUBLAS_OP_N; // B = V    : [K=total_len, N=HDIM]
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  
    }


    // 矩阵布局 row-major
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    // A: [M=seqlen, K=total_len], dtype=FP32
	cublasLtMatrixLayout_t Adesc;
	CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
		&Adesc, CUDA_R_32F,
		/* rows = M */ static_cast<int64_t>(seq_len),
		/* cols = K */ static_cast<int64_t>(total_len),
		/* ld   = K */ static_cast<int64_t>(nhead * total_len)));
	CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
		Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

	// B: [K=total_len, N=HDIM], dtype=CType
	cublasLtMatrixLayout_t Bdesc;
	CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
		&Bdesc, CType,
		/* rows = K */ static_cast<int64_t>(total_len),
		/* cols = N */ static_cast<int64_t>(HDIM),
		/* ld   = N */ static_cast<int64_t>(nkvhead * HDIM)));
	CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
		Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

	// D: [M=seqlen, N=HDIM], dtype=CType
	cublasLtMatrixLayout_t Ddesc;
	CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
		&Ddesc, CType,
		/* rows = M */ static_cast<int64_t>(seq_len),
		/* cols = N */ static_cast<int64_t>(HDIM),
		/* ld   = N */ static_cast<int64_t>(nhead * HDIM)));
	CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
		Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));


    // 算法选择
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspaceSize = 1 << 22; // 4MB
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize, sizeof(workspaceSize)));

    // RAII workspace
    auto workspace_storage =
        llaisys::core::context().runtime().allocateDeviceStorage(workspaceSize);
    void* workspace = static_cast<void*>(workspace_storage->memory()); // 允许为空，库会降级算法

	ASSERT(workspace_storage, "workspace empty!");

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, opDesc,
        Adesc, Bdesc, /*Cdesc*/ nullptr, Ddesc,
        preference, 1, &heuristic, &returnedResults));
    if (returnedResults == 0) {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Ddesc);
        cublasLtDestroy(ltHandle);
        throw std::runtime_error("cublasLtMatmul: no heuristic result");
    }

    // 5) loop over heads (GQA: nkvhead < nhead)
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // nhead nkvhead, qwen2中分别为12和2能够整除
    

    for (size_t h = 0; h < nhead; ++h) {
        const size_t kvh = (h * nkvhead) / nhead; // head -> kv_head

        // row-major 
        const float* A = score     + h * total_len;
        const T*     B = v         + kvh * HDIM;
        T*           D = atten_val + h * HDIM;

        CHECK_CUBLAS(cublasLtMatmul(
            ltHandle,
            opDesc,
            &alpha,
            A, Adesc,        // const float* -> const void*
            B, Bdesc,        // const T* -> const void* 
            &beta,
            D, Ddesc, 
            D, Ddesc,        // T* -> void*     
            &heuristic.algo,
            workspace, workspaceSize,
            stream_in));
    }
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaStreamSynchronize(stream_in));


	cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Ddesc);
	cublasLtMatmulDescDestroy(opDesc);
    cublasLtDestroy(ltHandle);

}


// 显式实例化
template void atten3d_hdim128_vproj_kernel<float, CUDA_R_32F>(
    float*, const float*, const float*, size_t, size_t, size_t, size_t, cudaStream_t);

template void atten3d_hdim128_vproj_kernel<__half, CUDA_R_16F>(
    __half*, const __half*, const float*, size_t, size_t, size_t, size_t, cudaStream_t);

template void atten3d_hdim128_vproj_kernel<__nv_bfloat16, CUDA_R_16BF>(
    __nv_bfloat16*, const __nv_bfloat16*, const float*, size_t, size_t, size_t, size_t, cudaStream_t);



} // namespace llaisys::ops::nvidia::kernels
