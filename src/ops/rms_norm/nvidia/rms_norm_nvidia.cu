#include "llaisys.h"
#include "../../../core/llaisys_core.hpp"

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cub/cub.cuh>

namespace {


template <typename T, unsigned BLOCK_SIZE>
__global__ void rmsnorm_kernel(T *out, const T *in, const T *weight, size_t n, size_t d, float eps) {
	// inv_rms = sqrt(block_sum_square / d + eps)
	// rms_norm(v_i) = v_i / inv_rms * w_i
	
	using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    size_t row = blockIdx.x;    
    size_t tid = threadIdx.x;

   	
    float thread_sum_square = 0.f;
    for (size_t col = tid; col < d; col += BLOCK_SIZE) {
        size_t index = row * d + col;
        float v;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) v = __bfloat162float(in[index]);
        else if constexpr (std::is_same_v<T, __half>)   v = __half2float(in[index]);
        else if constexpr (std::is_same_v<T, float>)    v = in[index];
        thread_sum_square += static_cast<double>(v) * static_cast<double>(v);
    }

    float block_sum_square = BlockReduce(temp_storage).Sum(thread_sum_square); 
    

	__shared__ float shared_inv_rms;
	if (threadIdx.x == 0) {
		shared_inv_rms = sqrtf(block_sum_square / d + eps);
	}
	__syncthreads();

	float inv_rms = shared_inv_rms;
    for (size_t col = tid; col < d; col += BLOCK_SIZE) {
        size_t index = row * d + col;
        float v, w;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
			v = __bfloat162float(in[index]);
			w = __bfloat162float(weight[col]);
		}
        else if constexpr (std::is_same_v<T, __half>) {
			v = __half2float(in[index]);
			w = __half2float(weight[col]);
		}  
        else if constexpr (std::is_same_v<T, float>) {
			v = in[index];
			w = weight[col];
		}		
        float rms_normed = (v / inv_rms) * w;

        if constexpr (std::is_same_v<T, __nv_bfloat16>) out[index] = __float2bfloat16(rms_normed);
        else if constexpr (std::is_same_v<T, __half>)   out[index] = __float2half(rms_normed);
        else if constexpr (std::is_same_v<T, float>)    out[index] = rms_normed;
    }
}

} // anomynous namespace


namespace llaisys::ops::nvidia {

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
			  size_t sequence_length, size_t embedding_dim, float eps) {
	
    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
	
	dim3 block(256), grid(static_cast<unsigned>(sequence_length));

	switch (type) {
		case LLAISYS_DTYPE_F32:
			rmsnorm_kernel<float, 256><<<grid, block, 0, s>>>(
				reinterpret_cast<float*>(out),
				reinterpret_cast<const float*>(in),
				reinterpret_cast<const float*>(weight),
				sequence_length,
			 	embedding_dim,
				eps);
     	 	break;
									   
			
		case LLAISYS_DTYPE_F16:
			rmsnorm_kernel< __half, 256><<<grid, block, 0, s>>>(
				reinterpret_cast< __half*>(out),
				reinterpret_cast<const  __half*>(in),
				reinterpret_cast<const  __half*>(weight),
				sequence_length,
			 	embedding_dim,
				eps);
     	 	break;
		case LLAISYS_DTYPE_BF16:
			rmsnorm_kernel<__nv_bfloat16, 256><<<grid, block, 0, s>>>(
				reinterpret_cast<__nv_bfloat16*>(out),
				reinterpret_cast<const __nv_bfloat16*>(in),
				reinterpret_cast<const __nv_bfloat16*>(weight),
				sequence_length,
			 	embedding_dim,
				eps);
     	 	break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }		


}

} // namespace llaisys::ops::nvidia