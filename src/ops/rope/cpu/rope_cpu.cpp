#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>


template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seqlen,
		   size_t nhead, size_t d) {
	for(size_t i = 0; i < seqlen; i++) {
		int64_t pos = pos_ids[i];
		for(size_t j = 0; j < nhead; j++) {
			for(size_t k = 0; k < d / 2; k++) {
				// inv_freq = 1.0 / freq = theta ^ (2 * k / d)
				// angle = pos * inv_freq;
				
				// if do this: float inv_freq  = 1.0 / std::pow(theta, (2.0f * k) / d);
				// may loss accuracy twice, once in 2.0f * k /d, once in pow.

				double exponent = -2.0 * k / d;           
				double inv_freq = std::pow(static_cast<double>(theta), exponent);
				float angle = static_cast<float>(pos * inv_freq);
				float cos_theta = std::cos(angle);
				float sin_theta = std::sin(angle);
				// calculate offset of a and b
				size_t a_offset = i * nhead * d + j * d + k;
				size_t b_offset = i * nhead * d + j * d + k + d / 2;

				// get a and b
				float a, b;
				if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
					a = llaisys::utils::cast<float>(in[a_offset]);
					b = llaisys::utils::cast<float>(in[b_offset]);
				} 
				else {
					a = in[a_offset];
					b = in[b_offset];
				}
				
				// calculate a_prime and b_prime
				float a_prime = a * cos_theta - b * sin_theta;
				float b_prime = a * sin_theta + b * cos_theta;

				// write to out matrix
				if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
					out[a_offset] = llaisys::utils::cast<T>(a_prime);
					out[b_offset] = llaisys::utils::cast<T>(b_prime);
				} 
				else { // float
					out[a_offset] = a_prime;
					out[b_offset] = b_prime;
				}
			}			
		}
	}
}

namespace llaisys::ops::cpu {
	void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
			  llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d) {	

		switch (type) {
			case LLAISYS_DTYPE_F32:
				return rope_(reinterpret_cast<float *>(out), 
							 reinterpret_cast<const float *>(in), 
							 reinterpret_cast<const int64_t *>(pos_ids),
							 theta, seqlen, nhead, d);
			case LLAISYS_DTYPE_BF16:
				return rope_(reinterpret_cast<llaisys::bf16_t *>(out), 
						     reinterpret_cast<const llaisys::bf16_t *>(in), 
							 reinterpret_cast<const int64_t *>(pos_ids),
							 theta, seqlen, nhead, d);
			case LLAISYS_DTYPE_F16:
				return rope_(reinterpret_cast<llaisys::fp16_t *>(out), 
							 reinterpret_cast<const llaisys::fp16_t *>(in), 
							 reinterpret_cast<const int64_t *>(pos_ids),
					 	     theta, seqlen, nhead, d);
			default:
				EXCEPTION_UNSUPPORTED_DATATYPE(type);

	 	}

	}
}