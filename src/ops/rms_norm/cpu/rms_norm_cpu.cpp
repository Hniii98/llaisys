#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t N, size_t D, float eps) {
	for (size_t i = 0; i < N; i++) {
		// caculate sum square of line N.
		float sum_sq = 0.0f;
		for(size_t j = 0; j < D; j++) {
			float val;
			if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            	val = llaisys::utils::cast<float>(in[i * D + j]);
			} else {
				val = in[i * D + j];
			}

			sum_sq += val * val;
		}
		// caculate rms of line N.
		float rms = std::sqrt(sum_sq / D + eps);


		// normalize element and mutiply weight
        for (size_t j = 0; j < D; ++j) {
            float val, scale;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[i * D + j]);
                scale = llaisys::utils::cast<float>(weight[j]);
            } else {
                val = static_cast<float>(in[i * D + j]);
                scale = static_cast<float>(weight[j]);
            }

            float normed = (val / rms) * scale;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * D + j] = llaisys::utils::cast<T>(normed);
            } else {
                out[i * D + j] = normed;
            }
        }
	}
}


namespace llaisys::ops::cpu {
	void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
				  size_t N, size_t D, float eps) {
		switch (type) {
			case LLAISYS_DTYPE_F32:
				return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight),
								 N, D, eps);
			case LLAISYS_DTYPE_BF16:
				return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight),
								 N, D, eps);
			case LLAISYS_DTYPE_F16:
				return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight),
								 N, D, eps);
    		default:
        		EXCEPTION_UNSUPPORTED_DATATYPE(type);
		}
	}
}