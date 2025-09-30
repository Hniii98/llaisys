#include "../../../../core/llaisys_core.hpp"
#include "../../../../device/nvidia/utils.cuh"
#include "atten3d_hdim.cuh"

#include <cudnn_frontend.h>
#include <cudnn.h>
#include <mutex>
#include <memory>
#include <atomic>  

namespace llaisys::ops::nvidia::kernels {

// 定义 CachedGraph 结构体
struct CachedGraph {
    cudnnHandle_t handle;
    cudnn_frontend::graph::Graph graph;
    int64_t workspace_size;
    std::shared_ptr<llaisys::core::Storage> workspace_buf;
    
    // 静态张量定义
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> tQ;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> tK;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> tV;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> tO_gen;
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
    cudaStream_t stream_in) {

    constexpr int HDIM = 128;
    
 
    static std::mutex cache_mutex;
    static std::unique_ptr<CachedGraph> cached_graph;
    static std::atomic<size_t> cached_total_len{0};
    
    // === 动态参数 ===
    constexpr int64_t B = 1;
    const int64_t S_q = static_cast<int64_t>(seq_len);
    const int64_t H_q = static_cast<int64_t>(nhead);
    const int64_t H_kv = static_cast<int64_t>(nkvhead);
    const int64_t Dm = HDIM;
    
    // === 动态步长计算 ===
    const int64_t q_strides[4] = {
        H_q * S_q * Dm,   // nhead * seq_len * 128
        Dm,               // 128
        H_q * Dm,         // nhead * 128
        1                 // 1
    };
    
    const int64_t o_strides[4] = {
        H_q * S_q * Dm,   // nhead * seq_len * 128 
        Dm,               // 128
        H_q * Dm,         // nhead * 128
        1                 // 1
    };

    // 检查是否需要重建图（原子操作）
    if (!cached_graph || total_len != cached_total_len.load()) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        // 双重检查
        if (!cached_graph || total_len != cached_total_len.load()) {
            // 销毁旧的
            if (cached_graph) {
                cudnnDestroy(cached_graph->handle);
                cached_graph.reset();
            }
            
            // 创建新的
            auto new_cached = std::make_unique<CachedGraph>();
            CHECK_CUDNN(cudnnCreate(&new_cached->handle));
            CHECK_CUDNN(cudnnSetStream(new_cached->handle, stream_in));
            
            // === 图构建 ===
            cudnn_frontend::graph::Graph graph;
            graph.set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
                 .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

            const int64_t S_kv = static_cast<int64_t>(total_len);
            
            // K和V的步长需要根据当前total_len计算
            const int64_t k_strides[4] = {
                H_kv * S_kv * Dm,  // nkvhead * total_len * 128
                Dm,                // 128
                H_kv * Dm,         // nkvhead * 128
                1                  // 1
            };
            
            const int64_t v_strides[4] = {
                H_kv * S_kv * Dm,  // nkvhead * total_len * 128
                Dm,                // 128
                H_kv * Dm,         // nkvhead * 128
                1                  // 1
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

            // 创建张量
            new_cached->tQ = make_tensor(
                std::vector<int64_t>{B, H_q, S_q, Dm}, 
                std::vector<int64_t>{q_strides[0], q_strides[1], q_strides[2], q_strides[3]}
            );
            new_cached->tK = make_tensor(
                std::vector<int64_t>{B, H_kv, S_kv, Dm},
                std::vector<int64_t>{k_strides[0], k_strides[1], k_strides[2], k_strides[3]}
            );
            new_cached->tV = make_tensor(
                std::vector<int64_t>{B, H_kv, S_kv, Dm},
                std::vector<int64_t>{v_strides[0], v_strides[1], v_strides[2], v_strides[3]}
            );

            // SDPA 配置
            cudnn_frontend::graph::SDPA_attributes sdpa_attr;
            sdpa_attr.set_attn_scale(scale);
            sdpa_attr.set_generate_stats(false);

            auto [tO_gen, tStats] = graph.sdpa(new_cached->tQ, new_cached->tK, new_cached->tV, sdpa_attr);

            // 输出张量
            new_cached->tO_gen = tO_gen;
            new_cached->tO_gen->set_dim(std::vector<int64_t>{B, H_q, S_q, Dm})
                   .set_stride(std::vector<int64_t>{o_strides[0], o_strides[1], o_strides[2], o_strides[3]})
                   .set_data_type(CudnnDType<T>::fe_type)
                   .set_output(true);

            // 构建图
            auto st_validate = graph.validate();  
            if (st_validate.is_bad()) { 
                cudnnDestroy(new_cached->handle);
                throw std::runtime_error(std::string("[FE] validate failed: ") + st_validate.get_message()); 
            }
            
            auto st_build = graph.build_operation_graph(new_cached->handle);   
            if (st_build.is_bad()) { 
                cudnnDestroy(new_cached->handle);
                throw std::runtime_error(std::string("[FE] build_operation_graph failed: ") + st_build.get_message()); 
            }
            
            auto st_heur = graph.create_execution_plans({cudnn_frontend::HeurMode_t::B});
            if (st_heur.is_bad()) { 
                cudnnDestroy(new_cached->handle);
                throw std::runtime_error(std::string("[FE] create_execution_plans failed: ") + st_heur.get_message()); 
            }
            
            auto st_plan = graph.build_plans();                   
            if (st_plan.is_bad()) { 
                cudnnDestroy(new_cached->handle);
                throw std::runtime_error(std::string("[FE] build_plans failed: ") + st_plan.get_message()); 
            }

            new_cached->graph = std::move(graph);
            new_cached->workspace_size = new_cached->graph.get_workspace_size();
            
            // 预分配workspace
            if (new_cached->workspace_size > 0) {
                new_cached->workspace_buf = llaisys::core::context().runtime()
                    .allocateDeviceStorage(static_cast<size_t>(new_cached->workspace_size));
            }
            
            cached_graph = std::move(new_cached);
            cached_total_len.store(total_len);  // 在锁内更新原子变量
            
            std::cout << "=== Rebuild, kvlen=" << total_len << " ===" << std::endl;
        }
    }
    
    // 执行阶段 
    void* workspace_ptr = nullptr;
    if (cached_graph->workspace_buf) {
        workspace_ptr = cached_graph->workspace_buf->memory();
    }
    
    // 更新stream
    CHECK_CUDNN(cudnnSetStream(cached_graph->handle, stream_in));
    
    // 设置tensor_map
    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensor_map;
    tensor_map.emplace(cached_graph->tQ, const_cast<T*>(q));
    tensor_map.emplace(cached_graph->tK, const_cast<T*>(k));
    tensor_map.emplace(cached_graph->tV, const_cast<T*>(v));
    tensor_map.emplace(cached_graph->tO_gen, static_cast<void*>(atten_val));

    auto st = cached_graph->graph.execute(cached_graph->handle, tensor_map, workspace_ptr);
    if (st.is_bad()) {
        throw std::runtime_error(std::string("[FE] execute failed: ") + st.get_message());
    }

    // 显式同步stream
    CHECK_CUDA(cudaStreamSynchronize(stream_in));
}

// 显式实例化
template void atten3d_hdim128_decode_kernel<__half>(
    __half*, const __half*, const __half*, const __half*,
    float, std::size_t, std::size_t, std::size_t, std::size_t, cudaStream_t);

template void atten3d_hdim128_decode_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    float, std::size_t, std::size_t, std::size_t, std::size_t, cudaStream_t);

} // namespace llaisys::ops::nvidia::kernels