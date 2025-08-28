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
	

	// template <typename T>
	// void linear_(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
	// 	/*
	// 		in: 				[N, D_in]
	// 		weight:     		[D_in, D_out]
	// 		out: 				[N, D_out]
	// 		bias:(optional):	[D_out]
	// 	*/
	// 	size_t N = in->shape()[0];
	// 	size_t D_in = in->shape()[1];
	// 	size_t D_out = weight->shape()[1];
	// 	size_t eloff = llaisys::utils::dsize(in->dtype()); // units: byte
	// 	ptrdiff_t in_stride_0 = in->strides()[0];
	// 	ptrdiff_t in_stride_1 = in->strides()[1];
	// 	ptrdiff_t w_stride_0 = weight->strides()[0];
	// 	ptrdiff_t w_stride_1 = weight->strides()[1];
	// 	ptrdiff_t o_stride_0 = out->strides()[0];
	// 	ptrdiff_t o_stride_1 = out->strides()[1];

	// 	for(size_t i = 0; i < N; i++) {
	// 		for(size_t j = 0; j < D_out; j++) {
	// 			T sum = llaisys::utils::cast<T>(0.0f);
	// 			for(size_t k = 0; k < D_in; k ++) {
	// 				T tmp = llaisys::utils::cast<T>(0.0f);
	// 				// tmp = in[i][k] * weight[k][j] 
	// 				// since input weight haven't transpose, so map it to weight[k][j]
	// 				mul(reinterpret_cast<std::byte *>(&tmp), 
	// 					in->data() + ((i * in_stride_0 + k * in_stride_1) * eloff), 
	// 					weight->data() + ((k * w_stride_0 + j * w_stride_1) * eloff),
	// 					out->dtype(),
	// 					1
	// 				);

	// 				// sum = sum + tmp
	// 				add(reinterpret_cast<std::byte *>(&sum), 
	// 					reinterpret_cast<const std::byte *>(&sum),
	// 					reinterpret_cast<const std::byte *>(&tmp),
	// 					out->dtype(),
	// 					1
	// 					);

	// 			}
	// 			// out[i][j] 
	// 			std::byte *pos = out->data() + ((i * o_stride_0 + j * o_stride_1) * eloff);
	// 			// pos = sum
	// 			std::memcpy(pos, &sum, eloff);
	// 			// pos = pos+ bias[j]
	// 			if(bias != nullptr) {
	// 				add(pos, pos, bias->data() + j * eloff, out->dtype(), 1);
	// 			}
	// 		}
	// 	}
	// }


	template <typename T>
void linear_(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    size_t N = in->shape()[0];
    size_t D_in = in->shape()[1];
    size_t D_out = weight->shape()[1];  

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
                    w_val  = llaisys::utils::cast<float>(weight_data[k * D_out + j]); 
                } else {
                    in_val = static_cast<float>(in_data[i * D_in + k]);
                    w_val  = static_cast<float>(weight_data[k * D_out + j]);
                }

                sum += in_val * w_val;
            }

            // 写结果
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out_data[i * D_out + j] = llaisys::utils::cast<T>(sum);
            } else {
                out_data[i * D_out + j] = static_cast<T>(sum);
            }

            // 加偏置
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