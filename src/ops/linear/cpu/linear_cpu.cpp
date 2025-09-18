#include "linear_cpu.hpp"

#include "../../../utils.hpp"
#include "../../add/cpu/add_cpu.hpp"

#include <cmath>
#include <cstring>

namespace {

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, 
			 size_t sequence_length, size_t embedding_dim, size_t features_dim) {

	for (size_t i = 0; i < sequence_length; i++) { 
		for (size_t j = 0; j < features_dim; j++) {
			float sum_temp = 0.0f; // all data type sum up in float
			for (size_t k = 0; k < embedding_dim; k++) {
				float in_val, w_val;

				if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
					in_val = llaisys::utils::cast<float>(in[i * embedding_dim + k]);
					w_val  = llaisys::utils::cast<float>(weight[j * embedding_dim + k]); // transpose by hand
				} else {
					in_val = static_cast<float>(in[i * embedding_dim + k]);
					w_val  = static_cast<float>(weight[j * embedding_dim + k]);         // transpose by hand
				}

				sum_temp += in_val * w_val;
			}

			if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
				out[i * features_dim + j] = llaisys::utils::cast<T>(sum_temp);
			} else {
				out[i * features_dim + j] = static_cast<T>(sum_temp);
			}

			if (bias) {
				if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
					float bias_val = llaisys::utils::cast<float>(bias[j]);
					float out_val  = llaisys::utils::cast<float>(out[i * features_dim + j]);
					out[i * features_dim + j] = llaisys::utils::cast<T>(out_val + bias_val);
				} else {
					out[i * features_dim + j] += bias[j];
				}
			}
		}
	}
}

template <typename T>
void linear_dispatch(std::byte *out, const std::byte *in, 
					 const std::byte *weight, const std::byte *bias,
					 size_t sequence_length, size_t embedding_dim, size_t features_dim) {
    linear_<T>(reinterpret_cast<T*>(out),
               reinterpret_cast<const T*>(in),
               reinterpret_cast<const T*>(weight),
               reinterpret_cast<const T*>(bias),
               sequence_length, 
			   embedding_dim, 
			   features_dim);
}


} // anomynous namespace


namespace llaisys::ops::cpu {


void linear(std::byte *out, const std::byte *in, 
			const std::byte *weight, const std::byte *bias, 
			size_t sequence_length, size_t embedding_dim, size_t features_dim, 
			llaisysDataType_t type) {
	/*
		in:      [sequence_length, embedding_dim]
		weight:  [features_dim, embedding_dim]
		out:     [sequence_length, features_dim]
		bias:    [features_dim] (optional)
	*/
	switch (type) {
		case LLAISYS_DTYPE_F32:  
			return linear_dispatch<float>(out, in, weight, bias, sequence_length, embedding_dim, features_dim);
		case LLAISYS_DTYPE_BF16: 
			return linear_dispatch<llaisys::bf16_t>(out, in, weight, bias, sequence_length, embedding_dim, features_dim);
		case LLAISYS_DTYPE_F16:  
			return linear_dispatch<llaisys::fp16_t>(out, in, weight, bias, sequence_length, embedding_dim, features_dim);
		default: 
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
	
}
} // namespace llaisys::ops::cpu