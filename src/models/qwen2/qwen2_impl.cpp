#include "qwen2_impl.hpp"

#include "../../tensor/tensor.hpp"
#include "../../ops/ops.hpp"
#include "llaisys/models/qwen2.h"

#include "llaisys_tensor_utils.hpp"   //  borrow/to_c_handle

#include <cmath>
#include <cstring>
#include <vector>

#include <chrono>

namespace llaisys::models::qwen2 {

//  权重数组分配/释放
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

//  helper
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

//  Qwen2Impl 
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

int64_t Qwen2Impl::forward(const int64_t* token_ids, size_t T, size_t pos_base) {
    std::cerr << "[Forward] start, T=" << T << " pos_base=" << pos_base << std::endl;

    if (!weights.in_embed || !weights.out_embed || !weights.out_norm_w) {
        std::cerr << "[Forward] ERROR: missing essential weights" << std::endl;
        return (T ? token_ids[T-1] : (meta.end_token >= 0 ? meta.end_token : 0));
    }

    auto in_embed  = llaisys::borrow(weights.in_embed);
    auto out_embed = llaisys::borrow(weights.out_embed);
    auto out_norm  = llaisys::borrow(weights.out_norm_w);
    auto dtype = meta.dtype;

    size_t V = safe_dim(in_embed, 0);
    size_t H = safe_dim(in_embed, 1);
    if (!V || !H) {
        std::cerr << "[Forward] ERROR: invalid in_embed dims" << std::endl;
        return (T ? token_ids[T-1] : (meta.end_token >= 0 ? meta.end_token : 0));
    }

    // 越界检查
    for (size_t i = 0; i < T; i++) {
        if (token_ids[i] < 0 || token_ids[i] >= (int64_t)V) {
            std::cerr << "[Forward] ERROR: token_ids[" << i << "]=" << token_ids[i]
                      << " out of range vocab=" << V << std::endl;
            return (meta.end_token >= 0 ? meta.end_token : 0);
        }
    }

    auto dev_type = in_embed->deviceType();
    auto dev_id   = in_embed->deviceId();

    // embedding
    auto indices = Tensor::create({T}, LLAISYS_DTYPE_I64, dev_type, dev_id);
    indices->load(token_ids);
    auto x = Tensor::create({T, H}, dtype, dev_type, dev_id);
    ops::embedding(x, indices, in_embed);

    auto pos_ids = Tensor::create({T}, LLAISYS_DTYPE_I64, dev_type, dev_id);
    
    {
        std::vector<int64_t> p(T);
        for (size_t i = 0; i < T; ++i) p[i] = static_cast<int64_t>(pos_base + i);
        pos_ids->load(p.data());
    }

    // blocks
    for (size_t l = 0; l < meta.nlayer; ++l) {
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
            std::cerr << "[Forward] missing weight(s) in layer " << l << ", stop" << std::endl;
            break;
        }

        // LN
        auto x_norm = Tensor::create({T, H}, dtype, dev_type, dev_id);
        ops::rms_norm(x_norm, x, attn_norm, meta.epsilon);

        // Q/K/V (仅针对本块 T 个 token)
        size_t Qd = safe_dim(q_w, 0);
        size_t Kd = safe_dim(k_w, 0);
        size_t Vd = safe_dim(v_w, 0);
        if (Qd != meta.nh * meta.dh || Kd != meta.nkvh * meta.dh || Vd != meta.nkvh * meta.dh) {
            std::cerr << "[Forward] layer " << l << " Q/K/V shape mismatch"
                      << " got (" << Qd << "," << Kd << "," << Vd << ")" << std::endl;
            break;
        }

        auto q2d = Tensor::create({T, (size_t)Qd}, dtype, dev_type, dev_id);
        auto k2d = Tensor::create({T, (size_t)Kd}, dtype, dev_type, dev_id);
        auto v2d = Tensor::create({T, (size_t)Vd}, dtype, dev_type, dev_id);

        ops::linear(q2d, x_norm, q_w, q_b);
        ops::linear(k2d, x_norm, k_w, k_b);
        ops::linear(v2d, x_norm, v_w, v_b);

        auto q = q2d->view({T, (size_t)meta.nh,   (size_t)meta.dh});
        auto k = k2d->view({T, (size_t)meta.nkvh, (size_t)meta.dh});
        auto v = v2d->view({T, (size_t)meta.nkvh, (size_t)meta.dh});

        // rope
        ops::rope(q, q, pos_ids, meta.theta);
        ops::rope(k, k, pos_ids, meta.theta);

        // 写入 KV 缓存
        if (use_kv_cache_) {
            append_kv(l, k, v, pos_base);
        }

        // 准备注意力的 K/V（历史 + 当前）
        tensor_t k_all = k;
        tensor_t v_all = v;
        if (use_kv_cache_) {
            size_t total_len = pos_base + T;                // 历史长度 + 当前块
            k_all = view_k_total(l, total_len);             // [total_len, nkvh, dh]
            v_all = view_v_total(l, total_len);
        }

        // 注意力：Q 只用本块，K/V 用全量
        auto attn3d = Tensor::create({T, (size_t)meta.nh, (size_t)meta.dh},
                                     dtype, dev_type, dev_id);
        float scale = (meta.dh > 0) ? (1.0f / std::sqrt(float(meta.dh))) : 1.0f;
        // 要求 ops::self_attention 支持 len_q != len_kv（大多数实现都支持）
        ops::self_attention(attn3d, q, k_all, v_all, scale);

        auto attn = attn3d->view({T, H});
        auto o = Tensor::create({T, H}, dtype, dev_type, dev_id);
        ops::linear(o, attn, o_w, nullptr);

        auto x_res1 = Tensor::create({T, H}, dtype, dev_type, dev_id);
        ops::add(x_res1, x, o);

        // MLP
        auto y_norm = Tensor::create({T, H}, dtype, dev_type, dev_id);
        ops::rms_norm(y_norm, x_res1, mlp_norm, meta.epsilon);

        size_t Id = safe_dim(gate_w, 0);
        if (Id != (size_t)meta.di) {
            std::cerr << "[Forward] layer " << l << " MLP dim mismatch, expect "
                      << meta.di << " got " << Id << std::endl;
            break;
        }

        auto gate = Tensor::create({T, Id}, dtype, dev_type, dev_id);
        auto up   = Tensor::create({T, Id}, dtype, dev_type, dev_id);
        ops::linear(gate, y_norm, gate_w, nullptr);
        ops::linear(up,   y_norm, up_w,   nullptr);

        auto act  = Tensor::create({T, Id}, dtype, dev_type, dev_id);
        ops::swiglu(act, gate, up);

        auto down = Tensor::create({T, H}, dtype, dev_type, dev_id);
        ops::linear(down, act, down_w, nullptr);

        auto x_res2 = Tensor::create({T, H}, dtype, dev_type, dev_id);
        ops::add(x_res2, x_res1, down);
        x = x_res2;
    }

    // 输出层（取最后一步）
    auto x_last = Tensor::create({T, H}, dtype, dev_type, dev_id);
    ops::rms_norm(x_last, x, out_norm, meta.epsilon);
    auto last_h = x_last->slice(0, T - 1, T); // [1,H]

    size_t Vocab = safe_dim(out_embed, 0);
    auto logits = Tensor::create({1, Vocab}, dtype, dev_type, dev_id);
    ops::linear(logits, last_h, out_embed, nullptr);

    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, dev_type, dev_id);
    auto max_val = Tensor::create({1}, dtype, dev_type, dev_id);
    ops::argmax(max_idx, max_val, logits);

    last_logits_ = logits;
    int64_t next_id = read_scalar_from_tensor<int64_t>(max_idx);
    std::cerr << "[Forward] done, next_id=" << next_id << std::endl;
    return next_id;
}

//  对外接口 
int64_t Qwen2Impl::prefill(const int64_t* token_ids, size_t T) {
  
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 覆盖上下文，清空缓存
    ctx_tokens_.assign(token_ids, token_ids + T);
    reset_cache();
    // prefill 一次性跑 T，并把 K/V 写入缓存（pos_base=0）
    int64_t next_id = forward(ctx_tokens_.data(), ctx_tokens_.size(), /*pos_base=*/0);
    // prefill 完成后，cache_len_ 已更新为 T
    
    // 结束计时并输出
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "[Prefill] Tokens: " << T << ", Time: " << duration.count() / 1000 << " ms " << std::endl;
    
    return next_id;
}

int64_t Qwen2Impl::decode_one(int64_t token_id) {
    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 这一步 token 的绝对位置
    size_t pos = ctx_tokens_.size();
    ctx_tokens_.push_back(token_id);
    // 只算这 1 个 token（pos_base=pos），并把它的 K/V 追加到缓存
    int64_t next_id = forward(&ctx_tokens_.back(), /*T=*/1, /*pos_base=*/pos);
    
    // 结束计时并输出
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "[Decode] Position: " << pos << ", Time: " << duration.count() / 1000 << " ms " << std::endl;
    
    return next_id;
}




void Qwen2Impl::reset_cache() {
    k_cache_.assign(meta.nlayer, nullptr);
    v_cache_.assign(meta.nlayer, nullptr);
    cache_len_ = 0;
}


void Qwen2Impl::ensure_layer_cache_capacity(size_t layer, size_t need_total_len,
                                            llaisysDataType_t dtype,
                                            llaisysDeviceType_t dev, int dev_id) {
    size_t nkvh = (size_t)meta.nkvh;
    size_t dh   = (size_t)meta.dh;

    auto grow = [&](tensor_t& buf) {
        size_t old_cap = buf ? buf->shape()[0] : 0;
        if (old_cap >= need_total_len) return;

        // 新容量：倍增或直接到 need_total_len
        size_t new_cap = old_cap ? old_cap : 1;
        while (new_cap < need_total_len) new_cap = new_cap * 2;

        // 新 tensor
        auto new_buf = Tensor::create({new_cap, nkvh, dh}, dtype, dev, dev_id);

        // 拷贝旧数据
        if (buf) {
            size_t elem_size = llaisys::utils::dsize(dtype);
            size_t to_copy = old_cap * nkvh * dh * elem_size;
            core::context().runtime().api()->memcpy_sync(
                new_buf->data(), buf->data(), to_copy,
                (dev == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_D2D : LLAISYS_MEMCPY_D2D
            );
        }
        buf = new_buf;
    };

    grow(k_cache_[layer]);
    grow(v_cache_[layer]);
}

void Qwen2Impl::append_kv(size_t layer, const tensor_t& k_t, const tensor_t& v_t, size_t pos_base) {
    // k_t/v_t: [T, nkvh, dh]
    auto dev_type = k_t->deviceType();
    auto dev_id   = k_t->deviceId();
    auto dtype    = k_t->dtype();

    size_t T     = k_t->shape()[0];
    size_t nkvh  = k_t->shape()[1];
    size_t dh    = k_t->shape()[2];
    size_t need_total_len = pos_base + T;

    ensure_layer_cache_capacity(layer, need_total_len, dtype, dev_type, dev_id);

    // 目标指针 = 缓存基址 + pos_base * (nkvh*dh)
    auto api = core::context().runtime().api();
    size_t elem_size = llaisys::utils::dsize(dtype);
    size_t row_elems = nkvh * dh;
    size_t row_bytes = row_elems * elem_size;

    // k
    {
        std::byte* dst_base = reinterpret_cast<std::byte*>(k_cache_[layer]->data());
        const std::byte* src_base = reinterpret_cast<const std::byte*>(k_t->data());
        for (size_t t = 0; t < T; ++t) {
            void* dst = dst_base + (pos_base + t) * row_bytes;
            const void* src = src_base + t * row_bytes;
            api->memcpy_sync(dst, src, row_bytes,
                (dev_type == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_D2D : LLAISYS_MEMCPY_D2D);
        }
    }
    // v
    {
        std::byte* dst_base = reinterpret_cast<std::byte*>(v_cache_[layer]->data());
        const std::byte* src_base = reinterpret_cast<const std::byte*>(v_t->data());
        for (size_t t = 0; t < T; ++t) {
            void* dst = dst_base + (pos_base + t) * row_bytes;
            const void* src = src_base + t * row_bytes;
            api->memcpy_sync(dst, src, row_bytes,
                (dev_type == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_D2D : LLAISYS_MEMCPY_D2D);
        }
    }

    // 更新缓存长度（最大位置+1）
    if (cache_len_ < need_total_len) cache_len_ = need_total_len;
}

tensor_t Qwen2Impl::view_k_total(size_t layer, size_t total_len) const {
    // 返回 [total_len, nkvh, dh] 的前缀 view（底层是 contiguous，直接 slice）
    return k_cache_[layer]->slice(0, 0, total_len);
}
tensor_t Qwen2Impl::view_v_total(size_t layer, size_t total_len) const {
    return v_cache_[layer]->slice(0, 0, total_len);
}

} // namespace llaisys::models::qwen2
