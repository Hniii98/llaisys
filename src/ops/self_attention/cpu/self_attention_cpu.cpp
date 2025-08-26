#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>	


template <typename T> 
void self_attention_(T *atten_val, const T *q, const T *k, const T *v, float scale, size_t seq_len, size_t nhead, 
					 size_t d, size_t total_len, size_t nkvhead, size_t dv) {
	
/*
  attn_val：[seqlen, nhead, dv]
           q：[seqlen, nhead, d]
           k：[total_len, nkvhead, d]
           v：[total_len, nkvhead, dv]
       scale：1 / sqrt(d)
*/
	
	std::vector<float> score(total_len);
	std::vector<float> weight(total_len);


	std::vector<float> acc(dv);


	size_t cache_len = total_len - seq_len;

	for(size_t query_i = 0; query_i < seq_len; query_i++) { 
		for(size_t  j = 0;  j < nhead; j++) {  
			
			size_t q_offset = query_i * nhead * d + j * d; // fetch a query
			const T *q_vec = q + q_offset;

			
			// size_t kv_group_size = nhead / nkvhead;
			// size_t kv_head_idx = j / kv_group_size;  map nhead to nkvhead index

			size_t kv_head_idx = j * nkvhead / nhead;


			//  1. score = Q * K ^ T * scale	
			float max_score = -std::numeric_limits<float>::infinity();
			size_t max_k_idx = query_i + cache_len;	
			// std::fill(score.begin(), score.end(), 0.0f);
			// std::fill(weight.begin(), weight.end(), 0.0f);


			for(size_t key_t = 0; key_t < total_len; key_t++) { 

				size_t k_offset = key_t * nkvhead * d + kv_head_idx * d;
				const T *k_vec = k + k_offset;  // off to line of key 
				
				if(key_t > max_k_idx) { // mask future
					score[key_t] = -std::numeric_limits<float>::infinity(); 
					continue;
				}

				float dot = 0.0f;
				for(size_t dim = 0; dim < d; dim++) {
					float qd, kd;

                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                        qd = llaisys::utils::cast<float>(q_vec[dim]);
                        kd = llaisys::utils::cast<float>(k_vec[dim]);
                    }
                    else {
                        qd = q_vec[dim];
                        kd = k_vec[dim];  
                    }
					dot += qd * kd;
				}

				float scaled = dot * scale;
				// remember the max scaled score
				if(key_t <= query_i + cache_len) max_score = std::max(max_score, scaled);
				score[key_t] = scaled;
			} 

		

			// 2. weight = casualsoftmax(score)

			// sum up exp of score in current timestep
			float exp_sum = 0.0f;
			for (size_t score_i = 0; score_i <= max_k_idx; score_i++) {
				float e_i = std::exp(score[score_i] - max_score);
				weight[score_i] = e_i;
				exp_sum += e_i;
			}
			ASSERT(exp_sum > 0.0, "Sum of exponentials should be greater than zero.");

			// do the softmax of score and store in weight vector
			for (size_t weight_i = 0; weight_i < total_len; weight_i++) {
				weight[weight_i] = (weight_i <= max_k_idx) ? 
								   (weight[weight_i] / exp_sum) : 
								    0.0f; // masked future
			}




			// 3. Y = weight * V
			size_t out_offset = query_i * nhead * dv + j * dv;
			T *out_vec = atten_val + out_offset;

			std::fill(acc.begin(), acc.end(), 0.0f);

            for (size_t weight_i = 0; weight_i < total_len; weight_i++) {
                const float w = weight[weight_i];
                if (w == 0.0f) continue;   // skip masked             

				size_t v_offset = weight_i * nkvhead * dv + kv_head_idx * dv;
                const T *v_vec = v + v_offset;


                for (size_t dim = 0; dim < dv; dim++) {
                    float vd;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                                  std::is_same_v<T, llaisys::fp16_t>) {
                        vd = llaisys::utils::cast<float>(v_vec[dim]);
                    } else {
                        vd = static_cast<float>(v_vec[dim]);
                    }
                    
                    acc[dim] += vd * w;
                }
            }


			for (size_t dim = 0; dim < dv; dim++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out_vec[dim] = llaisys::utils::cast<T>(acc[dim]);
                } else {
                    out_vec[dim] = static_cast<float>(acc[dim]);
                }
            }
		}
	}
}






namespace llaisys::ops::cpu {
	void self_attention(std::byte *atten_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, llaisysDataType_t type, 
						size_t seq_len, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
		
		
		switch (type) {
		case LLAISYS_DTYPE_F32:
			return self_attention_(reinterpret_cast<float *>(atten_val), 
								   reinterpret_cast<const float *>(q), 
								   reinterpret_cast<const float *>(k), 
								   reinterpret_cast<const float *>(v),
								   scale, seq_len, nhead, d, total_len, nkvhead, dv);
		case LLAISYS_DTYPE_BF16:
			return self_attention_(reinterpret_cast<llaisys::bf16_t *>(atten_val), 
								   reinterpret_cast<const llaisys::bf16_t *>(q), 
								   reinterpret_cast<const llaisys::bf16_t *>(k), 
								   reinterpret_cast<const llaisys::bf16_t *>(v),
								   scale, seq_len, nhead, d, total_len, nkvhead, dv);
		case LLAISYS_DTYPE_F16:
			return self_attention_(reinterpret_cast<llaisys::fp16_t *>(atten_val), 
								   reinterpret_cast<const llaisys::fp16_t *>(q), 
								   reinterpret_cast<const llaisys::fp16_t *>(k), 
								   reinterpret_cast<const llaisys::fp16_t *>(v),
								   scale, seq_len, nhead, d, total_len, nkvhead, dv);
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
