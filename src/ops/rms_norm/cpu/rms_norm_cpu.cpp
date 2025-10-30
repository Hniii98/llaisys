#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t sequence_length, size_t embedding_dim, float eps) {
	for (size_t i = 0; i < sequence_length; i++) {
		// caculate sum square of line sequence_length.
		float row_sum_square = 0.0f;
		for(size_t j = 0; j < embedding_dim; j++) {
			float val;
			if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            	val = llaisys::utils::cast<float>(in[i * embedding_dim + j]);
			} else {
				val = in[i * embedding_dim + j];
			}

			row_sum_square += val * val;
		}
		// caculate rms of line sequence_length.
		float rms = std::sqrt(row_sum_square / embedding_dim + eps);


		// normalize element and mutiply weight
        for (size_t j = 0; j < embedding_dim; ++j) {
            float val, scale;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[i * embedding_dim + j]);
                scale = llaisys::utils::cast<float>(weight[j]);
            } else {
                val = static_cast<float>(in[i * embedding_dim + j]);
                scale = static_cast<float>(weight[j]);
            }

            float rms_normed = (val / rms) * scale;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * embedding_dim + j] = llaisys::utils::cast<T>(rms_normed);
            } else {
                out[i * embedding_dim + j] = rms_normed;
            }
        }
	}
}


namespace llaisys::ops::cpu {
	void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
				  size_t sequence_length, size_t embedding_dim, float eps) {
		switch (type) {
			case LLAISYS_DTYPE_F32:
				return rms_norm_(reinterpret_cast<float *>(out), 
								 reinterpret_cast<const float *>(in), 
								 reinterpret_cast<const float *>(weight),
								 sequence_length, 
								 embedding_dim, 
								 eps);
			case LLAISYS_DTYPE_BF16:
				return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), 
								 reinterpret_cast<const llaisys::bf16_t *>(in), 
								 reinterpret_cast<const llaisys::bf16_t *>(weight),
								 sequence_length, 
								 embedding_dim, 
								 eps);
			case LLAISYS_DTYPE_F16:
				return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), 
								 reinterpret_cast<const llaisys::fp16_t *>(in), 
								 reinterpret_cast<const llaisys::fp16_t *>(weight),
								 sequence_length, 
								 embedding_dim, 
								 eps);
    		default:
        		EXCEPTION_UNSUPPORTED_DATATYPE(type);
		}
	}
}