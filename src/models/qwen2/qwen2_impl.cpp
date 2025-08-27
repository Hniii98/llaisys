#include "qwen2_impl.hpp"

#include <cstring>
#include <vector>
#include <cassert>




namespace llaisys::models::qwen2 {



Qwen2Impl::Qwen2Impl(const LlaisysQwen2Meta &meta)
    : meta(meta)
    , weights{}  
    , kv_key(meta.nlayer, nullptr)
    , kv_value(meta.nlayer, nullptr)
    , seq_len(0)
    , last_logits(nullptr) {
   
}



Qwen2Impl::~Qwen2Impl() {
    
}



Qwen2Impl* create(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
  
    return nullptr;
}


void destroy(Qwen2Impl *impl) {
   
}



LlaisysQwen2Weights* get_weights(Qwen2Impl *impl) {
   
    return nullptr;
}



int64_t infer(Qwen2Impl *impl, int64_t *token_ids, size_t ntoken) {
   return -1;
}



llaisysTensor_t get_logits(Qwen2Impl *impl) {
  
    return nullptr;
}

} // namespace llaisys::models::qwen2