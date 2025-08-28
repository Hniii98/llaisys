#pragma once

#include "llaisys/models/qwen2.h"
#include "tensor/tensor.hpp"

#include <vector>
#include <memory>

namespace llaisys::models::qwen2 {

struct Qwen2Impl {
    // 元信息
    LlaisysQwen2Meta meta;

    // 权重结构体（存放句柄，不负责销毁）
    LlaisysQwen2Weights weights{};

    // 设备信息
    llaisysDeviceType_t device;
    std::vector<int> device_ids;

    // 占位 logits
    llaisys::tensor_t logits;

    // 构造/析构
    Qwen2Impl(const LlaisysQwen2Meta& meta_,
              llaisysDeviceType_t device_,
              const int* device_ids_,
              int ndevice);

    ~Qwen2Impl();

    // 禁用拷贝，允许移动
    Qwen2Impl(const Qwen2Impl&) = delete;
    Qwen2Impl& operator=(const Qwen2Impl&) = delete;
    Qwen2Impl(Qwen2Impl&&) = default;
    Qwen2Impl& operator=(Qwen2Impl&&) = default;

    // 返回权重指针（C API 用）
    LlaisysQwen2Weights* getWeights() { return &weights; }

    // 推理接口（最小实现）
    int64_t prefill(const int64_t* token_ids, size_t ntoken);
    int64_t decode_one(int64_t token_id);

    // 返回 logits tensor
    llaisysTensor_t logits_tensor();
};

} // namespace llaisys::models::qwen2
