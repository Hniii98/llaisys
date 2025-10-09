#pragma once


#include "../../../../core/llaisys_core.hpp"
#include "../../../../device/nvidia/utils.cuh"

#include <cudnn_frontend.h>
#include <cudnn.h>
#include <memory>
#include <mutex>

namespace llaisys::ops::nvidia::kernels {

struct CachedGraph {
    // cuDNN FE
    cudnnHandle_t handle{};
    cudnn_frontend::graph::Graph graph;
    int64_t workspace_size{0};
    std::shared_ptr<llaisys::core::Storage> workspace_buf;

    // FE tensors
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> tQ;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> tK;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> tV;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> tO_gen;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> tBias;

    std::shared_ptr<llaisys::core::Storage> bias_buf;
    std::shared_ptr<llaisys::core::Storage> k_buf;
    std::shared_ptr<llaisys::core::Storage> v_buf;

    // cached capacities 
    int64_t cap_total_len{0};   // KV capacity 
    int64_t cap_nhead{0};    
    int64_t cap_nkvhead{0};   

};

template <typename T>
void atten3d_hdim128_decode_kernel(
    T *atten_val,  /* [seqlen, nhead, d=128] */
    const T *q,    /* [seqlen, nhead, d=128] */
    const T *k,    /* [total_len, nkvhead, d=128] */
    const T *v,    /* [total_len, nkvhead, d=128] */
    float scale,
    size_t seq_len,
    size_t nhead,
    size_t total_len,
    size_t nkvhead,
    cudaStream_t stream_in);

} // namespace llaisys::ops::nvidia::kernels
