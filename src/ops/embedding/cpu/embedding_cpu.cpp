#include "embedding_cpu.hpp"

#include "../../../utils.hpp"
#include <cstring>
#include "../../../llaisys/llaisys_tensor.hpp"
#include "llaisys.h"

namespace {

template <typename T>
void embedding_(T *out, 
				int64_t *index_list, size_t list_length, 
				const T *weight, size_t stride, size_t dsize) {

	for(size_t i = 0; i < list_length; i++) {
		size_t index =static_cast<size_t>(index_list[i]);
		size_t weight_offset = index * stride; // units: element
		size_t out_offset = i * stride; // units: element

		llaisys::core::context().runtime().api()->memcpy_sync(static_cast<void *>(out + out_offset),
															  static_cast<const void *>(weight + weight_offset),
															  dsize * stride,
															  LLAISYS_MEMCPY_H2H);

	}
	
}

}// anomynous namespace 



namespace llaisys::ops::cpu {

void embedding(std::byte *out, 
			   std::byte *index_list, size_t list_length, 
			   const std::byte *weight, size_t stride,
			   llaisysDataType_t type) {
	switch (type) {
		case LLAISYS_DTYPE_F32:	
			return embedding_<float>(reinterpret_cast<float *>(out), 
									 reinterpret_cast<int64_t *>(index_list), 
									 list_length,
									 reinterpret_cast<const float *>(weight),
								     stride,
									 llaisys::utils::dsize(type));

		case LLAISYS_DTYPE_BF16:
			return embedding_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), 
											   reinterpret_cast<int64_t *>(index_list), 
											   list_length,
											   reinterpret_cast<const llaisys::bf16_t *>(weight),
											   stride,
											   llaisys::utils::dsize(type));
			
		case LLAISYS_DTYPE_F16:
			return embedding_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), 
											   reinterpret_cast<int64_t *>(index_list), 
											   list_length,
											   reinterpret_cast<const llaisys::fp16_t *>(weight),
											   stride,
											   llaisys::utils::dsize(type));
		default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);	
	}
}

}// namespace llaisys::ops::cpu