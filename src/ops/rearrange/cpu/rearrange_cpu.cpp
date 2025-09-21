#include "rearrange_cpu.hpp"
#include "../../../utils.hpp"

namespace {

template <typename T>
void rearrange_(T *out, const T *in, const std::vector<size_t> &in_shape, const std::vector<ptrdiff_t> &in_stride) {
	const size_t ndim = in_shape.size();
	size_t nelmens = 1;
	for(auto s: in_shape) nelmens *= s;

	std::vector<size_t> index(ndim, 0);
	
	for(size_t i = 0; i < nelmens; i++) {
		
		ptrdiff_t flat_offset = 0;
		for(size_t d = 0; d < ndim; d++) {
			flat_offset += index[d] * in_stride[d];
		}

		out[i] = in[flat_offset];

		// 每次添加一个新元素后，更新index保证下一个元素的位置是正确
		// 累计进位器，当前位一旦溢出，置0当前位然后进位。
		for(int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
			if(++index[d] < in_shape[d])
				break;
			index[d] = 0; 
		 }
	}
}

} // anomynous namespace 





namespace llaisys::ops::cpu {
	void rearrange(std::byte *out,
				   std::byte *in, 
				   const std::vector<size_t> &input_shape,
				   const std::vector<ptrdiff_t> &input_strides,
				   llaisysDataType_t input_dtype)  {
	switch (input_dtype) {
		case LLAISYS_DTYPE_F32:
			return rearrange_(reinterpret_cast<float *>(out), 
							  reinterpret_cast<const float *>(in),
							  input_shape, 
							  input_strides);
		case LLAISYS_DTYPE_BF16:
			return rearrange_(reinterpret_cast<llaisys::bf16_t *>(out), 
							  reinterpret_cast<const llaisys::bf16_t *>(in),
							  input_shape, 
							  input_strides);
		case LLAISYS_DTYPE_F16:
			return rearrange_(reinterpret_cast<llaisys::fp16_t *>(out), 
							  reinterpret_cast<const llaisys::fp16_t *>(in),
							  input_shape, 
							  input_strides);
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(input_dtype);
		}
}
} // namespace llaisys::ops::cpu
