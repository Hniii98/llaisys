#include "rearrange_cpu.hpp"
#include "../../../utils.hpp"


template <typename T>
void rearrange_(T *out, const T *in, const std::vector<size_t> &in_shape, const std::vector<ptrdiff_t> &in_stride) {
	size_t ndim = in_shape.size();
	size_t nelmens = 1;
	for(auto s: in_shape) nelmens *= s;

	std::vector<size_t> index(ndim, 0);
	
	for(size_t i = 0; i < nelmens; i++) {
		
		size_t flat_offset = 0;
		for(size_t d = 0; d < ndim; d++) {
			flat_offset += in_stride[d] * in_shape[d];
		}

		out[i] = in[flat_offset];

		 // accumulate index array from right to left.
		 for(int d = ndim - 1; d >= 0; d--) {
			if(++index[d] < in_shape[d]) break;
			index[d] = 0; // carry in.
		 }

	}
}




namespace llaisys::ops::cpu {
	void rearrange(	std::byte *out,
					std::byte *in, 
					const std::vector<size_t> &in_shape,
					const std::vector<ptrdiff_t> &in_stride,
					llaisysDataType_t in_dtype)  {
	switch (in_dtype) {
    case LLAISYS_DTYPE_F32:
        return rearrange_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
						  in_shape, in_stride);
    case LLAISYS_DTYPE_BF16:
        return rearrange_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    	  in_shape, in_stride);
    case LLAISYS_DTYPE_F16:
        return rearrange_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                          in_shape, in_stride);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in_dtype);
    }
}
} // namespace llaisys::ops::cpu
