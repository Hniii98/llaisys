#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t n, size_t d) {
    for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < d; j++) {
			float gate_val, up_val;
			size_t offset = i * d + j;
        	if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
         	    gate_val = llaisys::utils::cast<float>(gate[offset]);  
				up_val = llaisys::utils::cast<float>(up[offset]);
        	} else {
				gate_val = gate[offset];
				up_val = up[offset];
        	}
			// sigmoid(x) = 1 / (1 + e^-x)
			// swish(x) = x * sigmoid(x)
			// swiglu(x) = up * swish(x)
			float sigmoid_val =  1.0f + expf(-gate_val);
			float swish_val = gate_val / sigmoid_val;
			float swiglu_val = up_val * swish_val ;
			
			if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
         	    out[offset] = llaisys::utils::cast<T>(swiglu_val);  
				
        	} else {
				out[offset] = swiglu_val;
			}		

		}	
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t n, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), n, d);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate),
                    reinterpret_cast<const llaisys::bf16_t *>(up), n, d);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate),
                    reinterpret_cast<const llaisys::fp16_t *>(up), n, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
