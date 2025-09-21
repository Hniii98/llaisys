#include "swiglu_nvidia.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../../device/nvidia/utils.cuh"


namespace {

template<typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t n, size_t d) {
	using llaisys::device::nvidia::utils::load_as_float;
	using llaisys::device::nvidia::utils::store_from_float;

	for(size_t seq_idx = blockIdx.x; seq_idx < n; seq_idx += gridDim.x) {
		for(size_t embed_idx = threadIdx.x; embed_idx < d; embed_idx += blockDim.x) {
			size_t offset =  seq_idx * d + embed_idx;
			
			float gate_val = load_as_float(gate + offset);
			float up_val = load_as_float(up + offset);

			float sigmoid_val = 1.0f + expf(-gate_val);
			float swish_val = gate_val / sigmoid_val;
			float swiglu_val = up_val * swish_val;

			store_from_float(out + offset, swiglu_val);
		}
	}
}


template <typename T>
void swiglu_dispatch(std::byte *out, const std::byte *gate, const std::byte *up, size_t n, size_t d) {
	dim3 block(256);
    dim3 grid(static_cast<unsigned>(n));
    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    swiglu_kernel<T><<<grid, block, 0, s>>>(
		reinterpret_cast<T *>(out),
    	reinterpret_cast<const T *>(gate),
		reinterpret_cast<const T *>(up),
		n,
		d);
}


} // anomynous namespace 


namespace llaisys::ops::nvidia {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type,
			size_t n, size_t d) {

	switch (type) {
		case LLAISYS_DTYPE_F32:
			swiglu_dispatch<float>(out, gate, up, n, d);
			break;
		case LLAISYS_DTYPE_F16:
			swiglu_dispatch<__half>(out, gate, up, n, d);
			break;
		case LLAISYS_DTYPE_BF16:
			swiglu_dispatch<__nv_bfloat16>(out, gate, up, n, d);
			break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);	
	}
}

} // namespace llaisys::ops::nvidia