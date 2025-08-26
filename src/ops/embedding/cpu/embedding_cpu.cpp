#include "embedding_cpu.hpp"

#include "../../../utils.hpp"
#include <cstring>

void embedding_(std::byte *dst, const std::byte *src, size_t numbyte) {
	std::memcpy(dst, src, numbyte);
}



namespace llaisys::ops::cpu {
void embedding(std::byte *dst, const std::byte *src, llaisysDataType_t type, size_t numel) {
	switch (type) {
		case LLAISYS_DTYPE_F32:	
			/* Pass through */
		case LLAISYS_DTYPE_BF16:
			/* Pass through */
		case LLAISYS_DTYPE_F16:
			return embedding_(dst, src, llaisys::utils::dsize(type) * numel);
		default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);	
	}
}

}// namespace llaisys::ops::cpu