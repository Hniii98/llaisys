#include "qwen2_impl.hpp"

#include "llaisys/models/qwen2.h"    
#include "../../llaisys/llaisys_tensor.hpp" 
#include <cassert>
#include <cstddef>

namespace llaisys::models::qwen2 {

// 分配/释放权重数组
static void alloc_weight_arrays(LlaisysQwen2Weights& w, size_t L) {
    w.attn_norm_w = new llaisysTensor_t[L];
    w.attn_q_w    = new llaisysTensor_t[L];
    w.attn_q_b    = new llaisysTensor_t[L];
    w.attn_k_w    = new llaisysTensor_t[L];
    w.attn_k_b    = new llaisysTensor_t[L];
    w.attn_v_w    = new llaisysTensor_t[L];
    w.attn_v_b    = new llaisysTensor_t[L];
    w.attn_o_w    = new llaisysTensor_t[L];
    w.mlp_norm_w  = new llaisysTensor_t[L];
    w.mlp_gate_w  = new llaisysTensor_t[L];
    w.mlp_up_w    = new llaisysTensor_t[L];
    w.mlp_down_w  = new llaisysTensor_t[L];

    for (size_t i = 0; i < L; ++i) {
        w.attn_norm_w[i] = nullptr;
        w.attn_q_w[i]    = nullptr;
        w.attn_q_b[i]    = nullptr;
        w.attn_k_w[i]    = nullptr;
        w.attn_k_b[i]    = nullptr;
        w.attn_v_w[i]    = nullptr;
        w.attn_v_b[i]    = nullptr;
        w.attn_o_w[i]    = nullptr;
        w.mlp_norm_w[i]  = nullptr;
        w.mlp_gate_w[i]  = nullptr;
        w.mlp_up_w[i]    = nullptr;
        w.mlp_down_w[i]  = nullptr;
    }

    w.in_embed   = nullptr;
    w.out_embed  = nullptr;
    w.out_norm_w = nullptr;
}

static void free_weight_arrays_keep_tensors(LlaisysQwen2Weights& w) {
    // 释放数组本体
    delete[] w.attn_norm_w; w.attn_norm_w = nullptr;
    delete[] w.attn_q_w;    w.attn_q_w    = nullptr;
    delete[] w.attn_q_b;    w.attn_q_b    = nullptr;
    delete[] w.attn_k_w;    w.attn_k_w    = nullptr;
    delete[] w.attn_k_b;    w.attn_k_b    = nullptr;
    delete[] w.attn_v_w;    w.attn_v_w    = nullptr;
    delete[] w.attn_v_b;    w.attn_v_b    = nullptr;
    delete[] w.attn_o_w;    w.attn_o_w    = nullptr;
    delete[] w.mlp_norm_w;  w.mlp_norm_w  = nullptr;
    delete[] w.mlp_gate_w;  w.mlp_gate_w  = nullptr;
    delete[] w.mlp_up_w;    w.mlp_up_w    = nullptr;
    delete[] w.mlp_down_w;  w.mlp_down_w  = nullptr;

    w.in_embed   = nullptr;
    w.out_embed  = nullptr;
    w.out_norm_w = nullptr;
}

static void free_all_weights_and_tensors(LlaisysQwen2Weights& w, size_t L) {
    auto destroy_if = [](llaisysTensor_t t) {
        if (t) tensorDestroy(t);
    };

    if (w.attn_norm_w) { for (size_t i = 0; i < L; ++i) destroy_if(w.attn_norm_w[i]); }
    if (w.attn_q_w)    { for (size_t i = 0; i < L; ++i) destroy_if(w.attn_q_w[i]); }
    if (w.attn_q_b)    { for (size_t i = 0; i < L; ++i) destroy_if(w.attn_q_b[i]); }
    if (w.attn_k_w)    { for (size_t i = 0; i < L; ++i) destroy_if(w.attn_k_w[i]); }
    if (w.attn_k_b)    { for (size_t i = 0; i < L; ++i) destroy_if(w.attn_k_b[i]); }
    if (w.attn_v_w)    { for (size_t i = 0; i < L; ++i) destroy_if(w.attn_v_w[i]); }
    if (w.attn_v_b)    { for (size_t i = 0; i < L; ++i) destroy_if(w.attn_v_b[i]); }
    if (w.attn_o_w)    { for (size_t i = 0; i < L; ++i) destroy_if(w.attn_o_w[i]); }
    if (w.mlp_norm_w)  { for (size_t i = 0; i < L; ++i) destroy_if(w.mlp_norm_w[i]); }
    if (w.mlp_gate_w)  { for (size_t i = 0; i < L; ++i) destroy_if(w.mlp_gate_w[i]); }
    if (w.mlp_up_w)    { for (size_t i = 0; i < L; ++i) destroy_if(w.mlp_up_w[i]); }
    if (w.mlp_down_w)  { for (size_t i = 0; i < L; ++i) destroy_if(w.mlp_down_w[i]); }

    destroy_if(w.in_embed);
    destroy_if(w.out_embed);
    destroy_if(w.out_norm_w);

    free_weight_arrays_keep_tensors(w);
}

Qwen2Impl::Qwen2Impl(const LlaisysQwen2Meta& meta_,
                     llaisysDeviceType_t device_,
                     const int* device_ids_,
                     int ndevice)
    : meta(meta_), device(device_) {

    if (device_ids_ && ndevice > 0) {
        device_ids.assign(device_ids_, device_ids_ + ndevice);
    }

    // 分配每层权重的槽位数组（置空），由 Python 侧填充
    alloc_weight_arrays(weights, meta.nlayer);

    // 最小实现：不分配 logits（logits_tensor 返回 nullptr 即可）
}

Qwen2Impl::~Qwen2Impl() {
    // 模型拥有权重，统一释放
    free_all_weights_and_tensors(weights, meta.nlayer);

    // logits 是 shared_ptr<Tensor>（tensor_t），自动回收
}

// ---- 推理占位实现 ----
int64_t Qwen2Impl::prefill(const int64_t* token_ids, size_t ntoken) {
    if (ntoken > 0) {
        // 返回输入的最后一个 token，模拟推理
        return token_ids[ntoken - 1];
    }
    return 0;
}

int64_t Qwen2Impl::decode_one(int64_t token_id) {
     static int counter = 0;
    counter++;
    if (counter > 5) {
        return meta.end_token;  // 模拟5步后结束
    }
    return token_id + 1;  // 模拟生成下一个 token
}

// ---- logits 暂不提供（返回 nullptr 即可）----
llaisysTensor_t Qwen2Impl::logits_tensor() {
    // 如需提供，可改成 new LlaisysTensor{ logits }，前提是 logits 有值
    return nullptr;
}

} // namespace llaisys::models::qwen2
