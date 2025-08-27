#pragma once

#include "qwen2.h"


#include "tensor/tensor.hpp" 
namespace llaisys::models::qwen2 {

struct Qwen2Impl {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    std::vector<llaisysTensor_t> kv_key, kv_value;
    size_t seq_len;
    llaisysTensor_t last_logits;

    Qwen2Impl(const LlaisysQwen2Meta &meta);
    ~Qwen2Impl();
};


Qwen2Impl* create(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
void destroy(Qwen2Impl *impl);
LlaisysQwen2Weights* get_weights(Qwen2Impl *impl);
int64_t infer(Qwen2Impl *impl, int64_t *token_ids, size_t ntoken);
llaisysTensor_t get_logits(Qwen2Impl *impl);

} // namespace llaisys::models::qwen2