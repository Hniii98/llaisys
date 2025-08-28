#include "cpu_linear.hpp"

#include "../../../utils.hpp"
#include "../../add/cpu/add_cpu.hpp"

#include <cmath>
#include <cstring>

/* Elementwise mutiply */
template <typename T>
void mul_(T *c, const T *a, const T *b, size_t numel) {
	for (size_t i = 0; i < numel; i++) {
		if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            c[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(a[i]) * llaisys::utils::cast<float>(b[i]));
        } else {
            c[i] = a[i] * b[i];
        }
	}
	
}


namespace llaisys::ops::cpu {
	
	void mul(std::byte *c, const std::byte *a, const std::byte	*b, llaisysDataType_t type, size_t numel) {
		switch (type) {
			case LLAISYS_DTYPE_F32:
        		return mul_(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
    		case LLAISYS_DTYPE_BF16:
        		return mul_(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
                    		reinterpret_cast<const llaisys::bf16_t *>(b), numel);
    		case LLAISYS_DTYPE_F16:
        		return mul_(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
                    		reinterpret_cast<const llaisys::fp16_t *>(b), numel);
    		default:
        		EXCEPTION_UNSUPPORTED_DATATYPE(type);
		}
		
    }

	template <typename T>
		void linear_(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
			/*
				in:      [N, D_in]
				weight:  [D_out, D_in]
				out:     [N, D_out]
				bias:    [D_out] (optional)
			*/
			size_t N = in->shape()[0];
			size_t D_in = in->shape()[1];
			size_t D_out = weight->shape()[0];  // 注意这里是 weight 的第 0 维

			const T* in_data = reinterpret_cast<const T*>(in->data());
			const T* weight_data = reinterpret_cast<const T*>(weight->data());
			T* out_data = reinterpret_cast<T*>(out->data());
			const T* bias_data = bias ? reinterpret_cast<const T*>(bias->data()) : nullptr;

			for (size_t i = 0; i < N; i++) {
				for (size_t j = 0; j < D_out; j++) {
					float sum = 0.0f;
					for (size_t k = 0; k < D_in; k++) {
						float in_val, w_val;

						if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
							in_val = llaisys::utils::cast<float>(in_data[i * D_in + k]);
							w_val  = llaisys::utils::cast<float>(weight_data[j * D_in + k]); // transpose by hand
						} else {
							in_val = static_cast<float>(in_data[i * D_in + k]);
							w_val  = static_cast<float>(weight_data[j * D_in + k]);         // transpose by hand
						}

						sum += in_val * w_val;
					}

					if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
						out_data[i * D_out + j] = llaisys::utils::cast<T>(sum);
					} else {
						out_data[i * D_out + j] = static_cast<T>(sum);
					}

					if (bias_data) {
						if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
							float bias_val = llaisys::utils::cast<float>(bias_data[j]);
							float out_val  = llaisys::utils::cast<float>(out_data[i * D_out + j]);
							out_data[i * D_out + j] = llaisys::utils::cast<T>(out_val + bias_val);
						} else {
							out_data[i * D_out + j] += bias_data[j];
						}
					}
				}
			}
		}



	
	void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
		// if (bias) {
		// 	std::cerr << "[CHECK][linear] in=(" << in->shape()[0] << "," << in->shape()[1] << ") "
		// 		<< "weight=(" << weight->shape()[0] << "," << weight->shape()[1] << ") "
		// 		<< "out=(" << out->shape()[0] << "," << out->shape()[1] << ")"
		// 		<< "bias=(" << bias->shape()[0] << "," << bias->shape()[1] << ")"
		// 		<< std::endl;
		// } else {
		// 	std::cerr << "[CHECK][linear] in=(" << in->shape()[0] << "," << in->shape()[1] << ") "
		// 		<< "weight=(" << weight->shape()[0] << "," << weight->shape()[1] << ") "
		// 		<< "out=(" << out->shape()[0] << "," << out->shape()[1] << ")"
		// 		<< std::endl;
		// }
		switch (in->dtype()) {
		case LLAISYS_DTYPE_F32:
			return linear_<float>(out, in, weight, bias);
    	case LLAISYS_DTYPE_BF16:
			return linear_<llaisys::bf16_t>(out, in, weight, bias);
    	case LLAISYS_DTYPE_F16:
        	return linear_<llaisys::fp16_t>(out, in, weight, bias);
    	default:
        	EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
			
		}
		
	}
}