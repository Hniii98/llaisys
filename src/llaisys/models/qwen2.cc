#include "qwen2.h"  
#include "ops.h"

#include "../../models/qwen2/qwen2_impl.hpp"
#include "../tensor/tensor.hpp"

__C {
   
    LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice
    ) {
      
        return (LlaisysQwen2Model*)llaisys::models::qwen2::create(meta, device, device_ids, ndevice);
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        llaisys::models::qwen2::destroy((llaisys::models::qwen2::Qwen2Impl*)model);
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        return llaisys::models::qwen2::get_weights((llaisys::models::qwen2::Qwen2Impl*)model);
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        return llaisys::models::qwen2::infer(
            (llaisys::models::qwen2::Qwen2Impl*)model,
            token_ids,
            ntoken
        );
    }


    int64_t llaisysQwen2ModelForwardOne(LlaisysQwen2Model *model, int64_t token_id) {
        return llaisysQwen2ModelInfer(model, &token_id, 1);
    }


    llaisysTensor_t llaisysQwen2ModelLogits(LlaisysQwen2Model *model) {
        return llaisys::models::qwen2::get_logits((llaisys::models::qwen2::Qwen2Impl*)model);
    }
}