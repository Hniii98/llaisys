#include "qwen2_impl.hpp"

#include "../../tensor/tensor.hpp"
#include "../../ops/ops.hpp"
#include "llaisys/models/qwen2.h"

#include "llaisys_tensor_utils.hpp"   //  borrow/to_c_handle

#include <cmath>
#include <cstring>
#include <vector>

namespace llaisys::models::qwen2 {

// -------- 权重数组分配/释放（不变） --------
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
        w.attn_norm_w[i] = nullptr; w.attn_q_w[i] = nullptr; w.attn_q_b[i] = nullptr;
        w.attn_k_w[i] = nullptr; w.attn_k_b[i] = nullptr; w.attn_v_w[i] = nullptr;
        w.attn_v_b[i] = nullptr; w.attn_o_w[i] = nullptr; w.mlp_norm_w[i] = nullptr;
        w.mlp_gate_w[i] = nullptr; w.mlp_up_w[i] = nullptr; w.mlp_down_w[i] = nullptr;
    }
    w.in_embed = w.out_embed = w.out_norm_w = nullptr;
}
static void free_weight_arrays_keep_tensors(LlaisysQwen2Weights& w) {
    delete[] w.attn_norm_w;  delete[] w.attn_q_w;   delete[] w.attn_q_b;
    delete[] w.attn_k_w;     delete[] w.attn_k_b;   delete[] w.attn_v_w;
    delete[] w.attn_v_b;     delete[] w.attn_o_w;   delete[] w.mlp_norm_w;
    delete[] w.mlp_gate_w;   delete[] w.mlp_up_w;   delete[] w.mlp_down_w;
    w.attn_norm_w = w.attn_q_w = w.attn_q_b = nullptr;
    w.attn_k_w = w.attn_k_b = w.attn_v_w = w.attn_v_b = nullptr;
    w.attn_o_w = w.mlp_norm_w = w.mlp_gate_w = w.mlp_up_w = w.mlp_down_w = nullptr;
    w.in_embed = w.out_embed = w.out_norm_w = nullptr;
}
static void free_all_weights_and_tensors(LlaisysQwen2Weights& w, size_t L) {
    auto destroy = [](llaisysTensor_t t){ if (t) tensorDestroy(t); };
    if (w.attn_norm_w) for (size_t i=0;i<L;++i) destroy(w.attn_norm_w[i]);
    if (w.attn_q_w)    for (size_t i=0;i<L;++i) destroy(w.attn_q_w[i]);
    if (w.attn_q_b)    for (size_t i=0;i<L;++i) destroy(w.attn_q_b[i]);
    if (w.attn_k_w)    for (size_t i=0;i<L;++i) destroy(w.attn_k_w[i]);
    if (w.attn_k_b)    for (size_t i=0;i<L;++i) destroy(w.attn_k_b[i]);
    if (w.attn_v_w)    for (size_t i=0;i<L;++i) destroy(w.attn_v_w[i]);
    if (w.attn_v_b)    for (size_t i=0;i<L;++i) destroy(w.attn_v_b[i]);
    if (w.attn_o_w)    for (size_t i=0;i<L;++i) destroy(w.attn_o_w[i]);
    if (w.mlp_norm_w)  for (size_t i=0;i<L;++i) destroy(w.mlp_norm_w[i]);
    if (w.mlp_gate_w)  for (size_t i=0;i<L;++i) destroy(w.mlp_gate_w[i]);
    if (w.mlp_up_w)    for (size_t i=0;i<L;++i) destroy(w.mlp_up_w[i]);
    if (w.mlp_down_w)  for (size_t i=0;i<L;++i) destroy(w.mlp_down_w[i]);
    destroy(w.in_embed); destroy(w.out_embed); destroy(w.out_norm_w);
    free_weight_arrays_keep_tensors(w);
}

// -------- 小工具 --------
template<typename T>
static T read_scalar_from_tensor(const tensor_t& t) {
    T v{};
    if (t->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(&v, t->data(), sizeof(T));
    } else {
        core::context().runtime().api()->memcpy_sync(&v, t->data(), sizeof(T), LLAISYS_MEMCPY_D2H);
    }
    return v;
}
static inline size_t safe_dim(const tensor_t& t, size_t i) {
    const auto& s = t->shape();
    return (i < s.size()) ? s[i] : 0;
}

// -------- Qwen2Impl --------
Qwen2Impl::Qwen2Impl(const LlaisysQwen2Meta& meta_,
                     llaisysDeviceType_t device_,
                     const int* device_ids_,
                     int ndevice)
    : meta(meta_), device(device_) {
    if (device_ids_ && ndevice > 0) device_ids.assign(device_ids_, device_ids_ + ndevice);
    alloc_weight_arrays(weights, meta.nlayer);
}
Qwen2Impl::~Qwen2Impl() {
    free_all_weights_and_tensors(weights, meta.nlayer);
}

// -------- Prefill：用现有算子搭一条最小前向（无 KV cache）--------
int64_t Qwen2Impl::prefill(const int64_t* token_ids, size_t T) {
    // 1) 基本权重检查
    if (!weights.in_embed || !weights.out_embed || !weights.out_norm_w) {
        return (T ? token_ids[T-1] : (meta.end_token >= 0 ? meta.end_token : 0));
    }

    // ⭐ C 句柄 → 非拥有 shared_ptr
    auto in_embed  = llaisys::borrow(weights.in_embed);
    auto out_embed = llaisys::borrow(weights.out_embed);
    auto out_norm  = llaisys::borrow(weights.out_norm_w);

    // 2) 维度/设备
    size_t V = safe_dim(in_embed, 0);
    size_t H = safe_dim(in_embed, 1);
    if (!V || !H) {
        return (T ? token_ids[T-1] : (meta.end_token >= 0 ? meta.end_token : 0));
    }
    auto dev_type = in_embed->deviceType();
    auto dev_id   = in_embed->deviceId();

    // 3) indices -> embedding
    auto indices = Tensor::create({T}, LLAISYS_DTYPE_I64, dev_type, dev_id);
    indices->load(token_ids);

    auto x = Tensor::create({T, H}, LLAISYS_DTYPE_F32, dev_type, dev_id);
    ops::embedding(x, indices, in_embed);

    // 4) pos_ids
    auto pos_ids = Tensor::create({T}, LLAISYS_DTYPE_I64, dev_type, dev_id);
    { std::vector<int64_t> p(T); for (size_t i=0;i<T;++i) p[i]=int64_t(i); pos_ids->load(p.data()); }

    // 5) L 层 block
    for (size_t l = 0; l < meta.nlayer; ++l) {
        // 需要的权重句柄 -> 非拥有 shared_ptr
        auto attn_norm = weights.attn_norm_w ? llaisys::borrow(weights.attn_norm_w[l]) : nullptr;
        auto q_w = weights.attn_q_w ? llaisys::borrow(weights.attn_q_w[l]) : nullptr;
        auto q_b = weights.attn_q_b ? llaisys::borrow(weights.attn_q_b[l]) : nullptr;
        auto k_w = weights.attn_k_w ? llaisys::borrow(weights.attn_k_w[l]) : nullptr;
        auto k_b = weights.attn_k_b ? llaisys::borrow(weights.attn_k_b[l]) : nullptr;
        auto v_w = weights.attn_v_w ? llaisys::borrow(weights.attn_v_w[l]) : nullptr;
        auto v_b = weights.attn_v_b ? llaisys::borrow(weights.attn_v_b[l]) : nullptr;
        auto o_w = weights.attn_o_w ? llaisys::borrow(weights.attn_o_w[l]) : nullptr;

        auto mlp_norm = weights.mlp_norm_w ? llaisys::borrow(weights.mlp_norm_w[l]) : nullptr;
        auto gate_w   = weights.mlp_gate_w ? llaisys::borrow(weights.mlp_gate_w[l]) : nullptr;
        auto up_w     = weights.mlp_up_w   ? llaisys::borrow(weights.mlp_up_w[l])   : nullptr;
        auto down_w   = weights.mlp_down_w ? llaisys::borrow(weights.mlp_down_w[l]) : nullptr;

        if (!attn_norm || !q_w || !k_w || !v_w || !o_w || !mlp_norm || !gate_w || !up_w || !down_w) {
            break; // 本层权重不全：跳过后续层
        }

        // LN
        auto x_norm = Tensor::create({T, H}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        ops::rms_norm(x_norm, x, attn_norm, meta.epsilon);

        // Q/K/V
        size_t Qd = safe_dim(q_w, 0), Kd = safe_dim(k_w, 0), Vd = safe_dim(v_w, 0);
        auto q = Tensor::create({T, Qd}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        auto k = Tensor::create({T, Kd}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        auto v = Tensor::create({T, Vd}, LLAISYS_DTYPE_F32, dev_type, dev_id);

        ops::linear(q, x_norm, q_w, q_b);
        ops::linear(k, x_norm, k_w, k_b);
        ops::linear(v, x_norm, v_w, v_b);

        // RoPE
        ops::rope(q, q, pos_ids, meta.theta);
        ops::rope(k, k, pos_ids, meta.theta);

        // Self-Attn
        auto attn = Tensor::create({T, H}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        float scale = (meta.dh > 0) ? (1.0f / std::sqrt(float(meta.dh))) : 1.0f;
        ops::self_attention(attn, q, k, v, scale);

        // 投影 + 残差
        auto o = Tensor::create({T, H}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        ops::linear(o, attn, o_w, nullptr);

        auto x_res1 = Tensor::create({T, H}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        ops::add(x_res1, x, o);

        // MLP：LN -> gate/up -> swiglu -> down -> 残差
        auto y_norm = Tensor::create({T, H}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        ops::rms_norm(y_norm, x_res1, mlp_norm, meta.epsilon);

        size_t Id = safe_dim(gate_w, 0);
        auto gate = Tensor::create({T, Id}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        auto up   = Tensor::create({T, Id}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        ops::linear(gate, y_norm, gate_w, nullptr);
        ops::linear(up,   y_norm, up_w,   nullptr);

        auto act  = Tensor::create({T, Id}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        ops::swiglu(act, gate, up);

        auto down = Tensor::create({T, H}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        ops::linear(down, act, down_w, nullptr);

        auto x_res2 = Tensor::create({T, H}, LLAISYS_DTYPE_F32, dev_type, dev_id);
        ops::add(x_res2, x_res1, down);

        x = x_res2; // 下一层
    }

    // 输出层：LN -> 取最后一步 -> out_proj -> argmax
    auto x_last = Tensor::create({T, H}, LLAISYS_DTYPE_F32, dev_type, dev_id);
    ops::rms_norm(x_last, x, out_norm, meta.epsilon);

    auto last_h = x_last->slice(0, T - 1, T); // [1,H]

    size_t Vocab = safe_dim(out_embed, 0);
    auto logits = Tensor::create({1, Vocab}, LLAISYS_DTYPE_F32, dev_type, dev_id);
    ops::linear(logits, last_h, out_embed, nullptr);

    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, dev_type, dev_id);
    auto max_val = Tensor::create({1}, LLAISYS_DTYPE_F32, dev_type, dev_id);
    ops::argmax(max_idx, max_val, logits);

    last_logits_ = logits; // 保存一份，C API 可通过 logits_tensor() 取句柄
    return read_scalar_from_tensor<int64_t>(max_idx);
}

int64_t Qwen2Impl::decode_one(int64_t token_id) {
    // 极简：直接把该 token 当作长度=1 的序列跑一次
    return prefill(&token_id, 1);
}

} // namespace llaisys::models::qwen2
