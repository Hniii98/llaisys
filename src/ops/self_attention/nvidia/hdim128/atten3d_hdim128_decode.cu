#include "../../../../core/llaisys_core.hpp"
#include "../../../../device/nvidia/utils.cuh"
#include "atten3d_hdim.cuh"

#include <cudnn_frontend.h>
#include <cudnn.h>


#include <vector>

namespace llaisys::ops::nvidia::kernels {

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
    cudaStream_t stream_in) {

    constexpr int HDIM = 128;

    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    CHECK_CUDNN(cudnnSetStream(handle, stream_in));
    

    cudnn_frontend::graph::Graph graph;
    graph.set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);


    const int64_t B    = 1;
    const int64_t S_q  = static_cast<int64_t>(seq_len);
    const int64_t H_q  = static_cast<int64_t>(nhead);
    const int64_t H_kv = static_cast<int64_t>(nkvhead);
    const int64_t S_kv = static_cast<int64_t>(total_len);
    const int64_t Dm   = HDIM;

    // 正确的步长设置：
    // cuDNN 维度顺序: [B, H, S, D]
    // 内存布局: [S, H, D]
    const int64_t q_strides[4] = {
        H_q * S_q * Dm,   // B 维步长：跳到下一个batch
        Dm,               // H 维步长：跳到下一个头
        H_q * Dm,         // S 维步长：跳到下一个序列位置
        1                 // D 维步长：跳到下一个维度元素
    };

    const int64_t k_strides[4] = {
        H_kv * S_kv * Dm,
        Dm,
        H_kv * Dm,
        1
    };

    const int64_t v_strides[4] = {
        H_kv * S_kv * Dm,
        Dm,
        H_kv * Dm,
        1
    };

    const int64_t o_strides[4] = {
        H_q * S_q * Dm,
        Dm,
        H_q * Dm,
        1
    };

    auto make_tensor = [&](const std::vector<int64_t>& dims,
                           const std::vector<int64_t>& strides) {
        return graph.tensor(
            cudnn_frontend::graph::Tensor_attributes()
                .set_dim(dims)
                .set_stride(strides)
                .set_data_type(CudnnDType<T>::fe_type)
        );
    };

    // 使用 cuDNN 需要的 [B, H, S, D] 维度顺序
    auto tQ = make_tensor(
        std::vector<int64_t>{B, H_q, S_q, Dm}, 
        std::vector<int64_t>{q_strides[0], q_strides[1], q_strides[2], q_strides[3]}
    );
    auto tK = make_tensor(
        std::vector<int64_t>{B, H_kv, S_kv, Dm},
        std::vector<int64_t>{k_strides[0], k_strides[1], k_strides[2], k_strides[3]}
    );
    auto tV = make_tensor(
        std::vector<int64_t>{B, H_kv, S_kv, Dm},
        std::vector<int64_t>{v_strides[0], v_strides[1], v_strides[2], v_strides[3]}
    );

    // SDPA 配置
    cudnn_frontend::graph::SDPA_attributes sdpa_attr;
    sdpa_attr.set_attn_scale(scale);
    sdpa_attr.set_generate_stats(false);

    auto [tO_gen, tStats] = graph.sdpa(tQ, tK, tV, sdpa_attr);

    // 输出也是 [B, H, S, D] 维度顺序
    tO_gen->set_dim(std::vector<int64_t>{B, H_q, S_q, Dm})
           .set_stride(std::vector<int64_t>{o_strides[0], o_strides[1], o_strides[2], o_strides[3]})
           .set_data_type(CudnnDType<T>::fe_type)
           .set_output(true);

    // 构图和执行代码保持不变...
    auto st_validate = graph.validate();  
    if (st_validate.is_bad()) { 
        CHECK_CUDNN(cudnnDestroy(handle)); 
        throw std::runtime_error(std::string("[FE] validate failed: ") + st_validate.get_message()); 
    }
    
    auto st_build = graph.build_operation_graph(handle);   
    if (st_build.is_bad()) { 
        CHECK_CUDNN(cudnnDestroy(handle)); 
        throw std::runtime_error(std::string("[FE] build_operation_graph failed: ") + st_build.get_message()); 
    }
    
    auto st_heur = graph.create_execution_plans({cudnn_frontend::HeurMode_t::B});
    if (st_heur.is_bad()) { 
        CHECK_CUDNN(cudnnDestroy(handle)); 
        throw std::runtime_error(std::string("[FE] create_execution_plans failed: ") + st_heur.get_message()); 
    }
    
    auto st_plan = graph.build_plans();                   
    if (st_plan.is_bad()) { 
        CHECK_CUDNN(cudnnDestroy(handle)); 
        throw std::runtime_error(std::string("[FE] build_plans failed: ") + st_plan.get_message()); 
    }

    // workspace & 执行
    int64_t workspace_bytes = graph.get_workspace_size();
    std::shared_ptr<llaisys::core::Storage> workspace_buf;
    void *workspace_ptr = nullptr;
    if (workspace_bytes > 0) {
        workspace_buf = llaisys::core::context().runtime().allocateDeviceStorage(static_cast<size_t>(workspace_bytes));
        workspace_ptr = workspace_buf ? workspace_buf->memory() : nullptr;    
    }

    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensor_map;
    tensor_map.emplace(tQ, const_cast<T*>(q));
    tensor_map.emplace(tK, const_cast<T*>(k));
    tensor_map.emplace(tV, const_cast<T*>(v));
    tensor_map.emplace(tO_gen, static_cast<void*>(atten_val));

    auto st = graph.execute(handle, tensor_map, workspace_ptr);
    if (st.is_bad()) {
        CHECK_CUDNN(cudnnDestroy(handle));
        throw std::runtime_error(std::string("[FE] execute failed: ") + st.get_message());
    }

    CHECK_CUDNN(cudnnDestroy(handle));
}


// 显式实例化， 避免在 .cu 文件里写了模板实现，却没把实例化符号导出到 .lib，
// 而调用端在另外一个翻译单元里显式实例化并链接，于是链接失败。
template void atten3d_hdim128_decode_kernel<__half>(
    __half*, const __half*, const __half*, const __half*,
    float, std::size_t, std::size_t, std::size_t, std::size_t, cudaStream_t);

template void atten3d_hdim128_decode_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    float, std::size_t, std::size_t, std::size_t, std::size_t, cudaStream_t);



} // namespace llaisys::ops::nvidia::kernels
