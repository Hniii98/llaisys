// src/llaisys/models/qwen2.cc
#include "llaisys/models/qwen2.h"
#include "../llaisys_tensor.hpp"            
#include "models/qwen2/qwen2_impl.hpp"        
#include "models/qwen2/llaisys_tensor_utils.hpp"

#include <memory>
#include <new>        
#include <cstdint>

__C {

    // Opaque C 侧 model 盒子：实际只持有一个 impl
    struct LlaisysQwen2Model {
        std::unique_ptr<llaisys::models::qwen2::Qwen2Impl> impl;
    };

    // 创建模型：转发到 Qwen2Impl 构造
    struct LlaisysQwen2Model* llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta* meta,
        llaisysDeviceType_t device,
        int* device_ids,
        int ndevice
    ) {
        try {
            auto m = new LlaisysQwen2Model;
            m->impl = std::make_unique<llaisys::models::qwen2::Qwen2Impl>(
                *meta, device, device_ids, ndevice
            );
            return m;
        } catch (const std::bad_alloc&) {
            return nullptr;
        } catch (...) {
            return nullptr;
        }
    }

    // 销毁模型：unique_ptr 先析构 impl（释放权重等），再 delete 外壳
    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model* model) {
        delete model;
    }

    // 借用权重指针：所有权在 model（impl）中，调用方**不要销毁**其中的 tensor
    struct LlaisysQwen2Weights* llaisysQwen2ModelWeights(struct LlaisysQwen2Model* model) {
        if (!model || !model->impl) return nullptr;
        return model->impl->getWeights();
    }

    // Prefill：返回下一 token（此处由 impl::prefill 决定；skeleton 中返回 end_token）
    int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model* model,
        int64_t* token_ids,
        size_t ntoken
    ) {
        if (!model || !model->impl) return -1;
        try {
            return model->impl->prefill(token_ids, ntoken);
        } catch (...) {
            return -1;
        }
    }

    // Decode one：返回下一 token id
    int64_t llaisysQwen2ModelForwardOne(
        struct LlaisysQwen2Model* model,
        int64_t token_id
    ) {
        if (!model || !model->impl) return -1;
        try {
            return model->impl->decode_one(token_id);
        } catch (...) {
            return -1;
        }
    }

    // 返回 logits tensor
    llaisysTensor_t llaisysQwen2ModelLogits(struct LlaisysQwen2Model* model) {
        if (!model || !model->impl) return nullptr;
        try {
            return model->impl->logits_tensor();
        } catch (...) {
            return nullptr;
        }
    }

}
