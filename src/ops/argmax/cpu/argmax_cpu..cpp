#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(T *max_idx, T *max_val, const T *vals, size_t numel) {
	// we assume vals is a 1D array for now.
	size_t idx = 0;
	T target = vals[0];

	for (size_t i = 1; i < numel; ++i) {

		if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
			if (llaisys::utils::cast<float>(vals[i]) >= llaisys::utils::cast<float>(target)) {
				target = vals[i];
				idx = i;
			}
		} else {
			if (vals[i] >= target) { // record latest one
				target = vals[i];
				idx = i;
			}
		}
		
	}
	
	max_idx[0] = llaisys::utils::cast<T>(idx);
	max_val[0] = target;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return argmax_(reinterpret_cast<float *>(max_idx), reinterpret_cast<float *>(max_val),
					   reinterpret_cast<const float *>(vals), numel);
	case LLAISYS_DTYPE_BF16:
		return argmax_(reinterpret_cast<llaisys::bf16_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
					   reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
	case LLAISYS_DTYPE_F16:
		return argmax_(reinterpret_cast<llaisys::fp16_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
					   reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
	default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}

}
} // namespace llaisys::ops::cpu
