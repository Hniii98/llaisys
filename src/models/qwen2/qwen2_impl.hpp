#pragma once

#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include "../../ops/ops.hpp"
#include "llaisys_tensor_utils.hpp"

#include <vector>

namespace llaisys::models::qwen2 {

struct Qwen2Impl {
    // ---- 模型元信息 ----
    LlaisysQwen2Meta    meta;
    // ---- 权重 ----
    // 注意：weights 里的 tensor 句柄由模型创建 & 销毁，不要在外部 delete
    LlaisysQwen2Weights weights{};


    // ---- 设备信息 ----
    llaisysDeviceType_t device;
    std::vector<int>    device_ids;

    // ---- 最近一次 prefill 结果的 logits ----
    // 用 shared_ptr<Tensor> 持有，保证生命周期
    llaisys::tensor_t   last_logits_{};

    // ---- 构造 & 析构 ----
    Qwen2Impl(const LlaisysQwen2Meta& meta_,
              llaisysDeviceType_t device_,
              const int* device_ids_,
              int ndevice);
    ~Qwen2Impl();

    Qwen2Impl(const Qwen2Impl&) = delete;
    Qwen2Impl& operator=(const Qwen2Impl&) = delete;
    Qwen2Impl(Qwen2Impl&&) = default;
    Qwen2Impl& operator=(Qwen2Impl&&) = default;

    // ---- 权重访问 ----
    LlaisysQwen2Weights* getWeights() { return &weights; }

    // ---- 推理接口 ----
     int64_t forward(const int64_t* token_ids, size_t T, size_t pos_base);
    int64_t prefill(const int64_t* token_ids, size_t ntoken);
    int64_t decode_one(int64_t token_id);

    // ---- Logits ----
    // C API 用：借用 shared_ptr（不转移所有权）
    llaisysTensor_t logits_tensor() {
         return llaisys::to_c_handle(last_logits_);
    }

private:
    // ---- KV cache ----
    void reset_cache();
    void ensure_layer_cache_capacity(size_t layer, size_t need_total_len, llaisysDataType_t dtype, llaisysDeviceType_t dev, int dev_id);
    void append_kv(size_t layer, const tensor_t& k_t, const tensor_t& v_t, size_t pos_base);
    tensor_t view_k_total(size_t layer, size_t total_len) const;
    tensor_t view_v_total(size_t layer, size_t total_len) const;

    // 上下文 token
    std::vector<int64_t> ctx_tokens_;

    // 缓存状态
    bool   use_kv_cache_ = true;   // 默认开启
    size_t cache_len_    = 0;      // 已缓存长度（== 当前上下文长度）
    // 每层一对缓存，shape 动态为 [cap, nkvh, dh]
    std::vector<tensor_t> k_cache_;
    std::vector<tensor_t> v_cache_;
    
};

} // namespace llaisys::models::qwen2
