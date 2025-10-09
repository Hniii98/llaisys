#include "rope_nvidia.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../../device/nvidia/utils.cuh"


namespace {

// TODO: divide pow(theta,−2k/d) into exp((−2lntheta)/d​)*k)
template<typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta, 
				size_t sequence_length, size_t num_heads, size_t embedding_dim) {
	using llaisys::device::nvidia::utils::load_as_float;
	using llaisys::device::nvidia::utils::store_from_float;
	// since gridDim.x can be very large, we ignore it's stride here.
	// gridDim.x could be used as sequence length safely. 			
	size_t sequence_index = blockIdx.x;
	if (sequence_index >= sequence_length) return;

	int64_t pos = pos_ids[sequence_index];

	for(size_t nhead = blockIdx.y; nhead < num_heads; nhead += gridDim.y) {	
		for(size_t dim = threadIdx.x; dim < embedding_dim / 2; dim += blockDim.x) {
			double exponent = -2.0 * static_cast<double>(dim) / static_cast<double>(embedding_dim);
			double inv_freq = pow(static_cast<double>(theta), exponent);

			// !! we assume the LLM context smaller that 2^24, so we can use int64_t to float safely.
			float angle = static_cast<float>(static_cast<double>(pos) * inv_freq);
			float cos_angle = cosf(angle);
			float sin_angle = sinf(angle);

			size_t a_offset = sequence_index * num_heads * embedding_dim + nhead * embedding_dim + dim;
			size_t b_offset = a_offset +  embedding_dim / 2;
			
			// get a and b
			float a = load_as_float(in + a_offset);
            float b = load_as_float(in + b_offset);  
			
			// caculate a' and b'
			float a_prime = a * cos_angle - b * sin_angle;
			float b_prime = a * sin_angle + b * cos_angle;
			
			// store a' and b'
			store_from_float(out + a_offset, a_prime);
            store_from_float(out + b_offset, b_prime);
		}
	}
}


template <typename T>
void rope_dispatch(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
				   size_t sequence_length, size_t num_heads, size_t embedding_dim) {
	dim3 block(256);
    dim3 grid(static_cast<unsigned>(sequence_length),
              static_cast<unsigned>(num_heads));
    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    rope_kernel<T><<<grid, block, 0, s>>>(
		reinterpret_cast<T*>(out),
    	reinterpret_cast<const T*>(in),
        reinterpret_cast<const int64_t*>(pos_ids),
        theta,
        sequence_length, 
		num_heads, 
		embedding_dim);
}

} // anomynous namespace 


namespace llaisys::ops::nvidia {

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
		  llaisysDataType_t type, size_t sequence_length, size_t num_heads, size_t embedding_dim) {

	switch (type) {
		case LLAISYS_DTYPE_F32:
			rope_dispatch<float>(out, in, pos_ids, theta, sequence_length, num_heads, embedding_dim);
			break;
		case LLAISYS_DTYPE_F16:
			rope_dispatch<__half>(out, in, pos_ids, theta, sequence_length, num_heads, embedding_dim);
			break;
		case LLAISYS_DTYPE_BF16:
			rope_dispatch<__nv_bfloat16>(out, in, pos_ids, theta, sequence_length, num_heads, embedding_dim);
			break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);			
	}
}

} // namespace llaisys::ops::nvidia