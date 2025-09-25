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
	
	std::vector<float> score(total_len); // score result for each query
	std::vector<float> weight(total_len);
	std::vector<float> acc(dv);

	size_t cache_len = total_len - seq_len;

	for(size_t query_i = 0; query_i < seq_len; query_i++) { 
		for(size_t  head_i = 0;  head_i < nhead; head_i++) {  
			// 找到第i个query的行起始指针
			size_t q_offset = query_i * nhead * d + head_i * d; // fetch a query
			const T *q_vec = q + q_offset; // get the start pointer of a row in a query
			size_t kv_head_idx = head_i * nkvhead / nhead; // map query to kv head group

			//  1. score = Q * K ^ T * scale	
			float max_score = -std::numeric_limits<float>::infinity();
			size_t max_k_idx = query_i + cache_len;	

			// 当前query对所有的key做矩阵乘得到total_len长度的score
			// 同时将未来的key对应的score设为-inf
			for(size_t key_i = 0; key_i < total_len; key_i++) { 

				size_t k_offset = key_i * nkvhead * d + kv_head_idx * d;
				const T *k_vec = k + k_offset;  // off to line of key 

				//  key_i:[0, 1, ..., max_k_idx, ... , total_len-1]
				// score for query_i:
				//        [                 ..., -inf,        -inf]

				if(key_i > max_k_idx) { // mask future for query_i
					score[key_i] = -std::numeric_limits<float>::infinity(); 
					continue;
				}
				
				// query_i 和 key_i 计算dot product
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
				// remember the max scaled score for later softmax
				if(key_i <= query_i + cache_len) max_score = std::max(max_score, scaled);
				score[key_i] = scaled;
			} 

		

			// 2. weight = casualsoftmax(score)

			// 计算query_i下，对total_len个key的score结果做exponent sum
			// future部分的key e_i设为0，同时为了数组稳定性，score会先减去
			// 之前记录的max_score，得到维度维total_len的最终权重
			float exp_sum = 0.0f;
			// caculate exp sum of valid number in score vector of query_i
			for (size_t score_i = 0; score_i <= max_k_idx; score_i++) {
				float e_i = std::exp(score[score_i] - max_score);
				// to avoid overflow, value of score_i minus max_score first.
				if (std::isnan(e_i) || std::isinf(e_i)) e_i = 0.0f;
				weight[score_i] = e_i;
				exp_sum += e_i;
			}

			if (exp_sum <= 0.0f) {
				std::fill(weight.begin(), weight.end(), 0.0f);
				continue;
			}

			ASSERT(exp_sum > 0.0, "Sum of exponentials should be greater than zero.");

			// divide e_i by exp sum of score
			for (size_t weight_i = 0; weight_i < total_len; weight_i++) {
				weight[weight_i] = (weight_i <= max_k_idx) ? 
								   (weight[weight_i] / exp_sum) : 
								    0.0f; // masked future
			}

			// 3. Y = weight * V
			// 每个out_idx对应一个权重weight_i, 然后在dv上广播。
			size_t out_offset = query_i * nhead * dv + head_i * dv;
			T *out_vec = atten_val + out_offset;

			std::fill(acc.begin(), acc.end(), 0.0f); // clear acc
			// calculate vi for qi.
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
