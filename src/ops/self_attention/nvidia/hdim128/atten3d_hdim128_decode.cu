#include "../../../../core/llaisys_core.hpp"
#include "../../../../device/nvidia/utils.cuh"
#include "atten3d_hdim128.cuh"
#include "atten3d_hdim128_decode.cuh"

#include <unordered_map>
#include <cudnn_frontend.h>
#include <cudnn.h>
#include <mutex>
#include <memory> 
#include <cuda_runtime.h>
#include <math_constants.h>

namespace llaisys::ops::nvidia::kernels{
    
__global__ void fill_neg_inf(float *p, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = -CUDART_INF_F;
}

template <typename T>
std::unique_ptr<CachedGraph> create_graph(
    int64_t batch,
    int64_t seq_len,
    int64_t nhead,
    int64_t total_len,
    int64_t nkvhead,
    int64_t hidden_dims,
    float scale,
    cudaStream_t stream) {

    auto cached_graph = std::make_unique<CachedGraph>();
    CHECK_CUDNN(cudnnCreate(&cached_graph->handle));
    CHECK_CUDNN(cudnnSetStream(cached_graph->handle, stream));

    // strides array
    const int64_t o_strides[4] = {nhead * seq_len * hidden_dims, hidden_dims, nhead * hidden_dims, 1};
    const int64_t q_strides[4] = {nhead * seq_len * hidden_dims, hidden_dims, nhead * hidden_dims, 1};
    const int64_t k_strides[4] = {nkvhead * total_len * hidden_dims, hidden_dims, nkvhead * hidden_dims, 1};
    const int64_t v_strides[4] = {nkvhead * total_len * hidden_dims, hidden_dims, nkvhead * hidden_dims, 1};
 

    cudnn_frontend::graph::Graph graph;
    graph.set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto make_tensor = [&](const std::vector<int64_t>& dims,
                           const std::vector<int64_t>& strides,
                           cudnn_frontend::DataType_t type) {
        return graph.tensor(
            cudnn_frontend::graph::Tensor_attributes()
                .set_dim(dims)
                .set_stride(strides)
                .set_data_type(type)
        );
    };

    cached_graph->tQ = make_tensor({batch, nhead, seq_len, hidden_dims},
                         {q_strides[0], q_strides[1], q_strides[2], q_strides[3]},
                         CudnnDType<T>::fe_type);

    cached_graph->tK = make_tensor({batch, nkvhead, total_len, hidden_dims},
                         {k_strides[0], k_strides[1], k_strides[2], k_strides[3]},
                         CudnnDType<T>::fe_type);

    cached_graph->tV = make_tensor({batch, nkvhead, total_len, hidden_dims},
                         {v_strides[0], v_strides[1], v_strides[2], v_strides[3]},
                         CudnnDType<T>::fe_type);

    cached_graph->tBias = make_tensor({1, 1, 1, total_len},
                            {0, 0, 0, 1},
                            cudnn_frontend::DataType_t::FLOAT);

    cudnn_frontend::graph::SDPA_attributes sdpa_attr;
    sdpa_attr.set_attn_scale(scale);
    sdpa_attr.set_generate_stats(false);
    sdpa_attr.set_bias(cached_graph->tBias);

    auto [tO_gen, tStats] = graph.sdpa(cached_graph->tQ, cached_graph->tK, cached_graph->tV, sdpa_attr);

    cached_graph->tO_gen = tO_gen;
    cached_graph->tO_gen->set_dim({batch, nhead, seq_len, hidden_dims})
               .set_stride({o_strides[0], o_strides[1], o_strides[2], o_strides[3]})
               .set_data_type(CudnnDType<T>::fe_type)
               .set_output(true);

    auto st_validate = graph.validate();
    if (st_validate.is_bad()) {
        cudnnDestroy(cached_graph->handle);
        throw std::runtime_error(std::string("[FE] validate failed: ") + st_validate.get_message());
    }

    auto st_build = graph.build_operation_graph(cached_graph->handle);
    if (st_build.is_bad()) {
        cudnnDestroy(cached_graph->handle);
        throw std::runtime_error(std::string("[FE] build_operation_graph failed: ") + st_build.get_message());
    }

    auto st_heur = graph.create_execution_plans({cudnn_frontend::HeurMode_t::B});
    if (st_heur.is_bad()) {
        cudnnDestroy(cached_graph->handle);
        throw std::runtime_error(std::string("[FE] create_execution_plans failed: ") + st_heur.get_message());
    }

    auto st_plan = graph.build_plans();
    if (st_plan.is_bad()) {
        cudnnDestroy(cached_graph->handle);
        throw std::runtime_error(std::string("[FE] build_plans failed: ") + st_plan.get_message());
    }

    cached_graph->graph = std::move(graph);
    cached_graph->workspace_size = cached_graph->graph.get_workspace_size();
    if (cached_graph->workspace_size > 0) {
        cached_graph->workspace_buf = llaisys::core::context().runtime()
            .allocateDeviceStorage(static_cast<size_t>(cached_graph->workspace_size));
    }

    cached_graph->cap_total_len = total_len;
    cached_graph->cap_nhead  = nhead;
    cached_graph->cap_nkvhead = nkvhead;

    const size_t bias_bytes = static_cast<size_t>(total_len) * sizeof(float);
    const size_t kv_elements   = static_cast<size_t>(batch) * static_cast<size_t>(nkvhead) * 
                                 static_cast<size_t>(total_len) * static_cast<size_t>(hidden_dims);
    const size_t kv_bytes   = kv_elements * sizeof(T);

    cached_graph->bias_buf = llaisys::core::context().runtime().allocateDeviceStorage(bias_bytes);
    cached_graph->k_buf    = llaisys::core::context().runtime().allocateDeviceStorage(kv_bytes);
    cached_graph->v_buf    = llaisys::core::context().runtime().allocateDeviceStorage(kv_bytes);

    return cached_graph;
}


// 这里的图执行暂时不支持prefix_len回退的场景，简单支持decode时prefix单调增长的情形。
template <typename T>
void execute_graph(
    CachedGraph* cached_graph,
    T*           atten_val,
    const T*     q,
    const T*     k,
    const T*     v,
    size_t       prefix_len,   // 实际使用的 KV 长度（前缀长度）
    cudaStream_t stream) {

    // 让 cuDNN FE 在同一条 stream 上执行
    CHECK_CUDNN(cudnnSetStream(cached_graph->handle, stream));

    // 从缓存能力里取形状信息
    const int64_t kv_capacity   = cached_graph->cap_total_len;   // 预分配 KV 上限
    const int64_t kv_num_heads  = cached_graph->cap_nkvhead;     // KV 头数
    constexpr int64_t hidden_dim = 128;                           // 固定 128

    // K/V 预处理：拷贝前缀 + 清零尾部
    // 前缀 [0, prefix_len) 从真实 K/V 拷贝到 staging 缓冲
    if (prefix_len > 0) {
        const size_t prefix_elems = static_cast<size_t>(prefix_len) *
                                    static_cast<size_t>(kv_num_heads) *
                                    static_cast<size_t>(hidden_dim);
        const size_t prefix_bytes = prefix_elems * sizeof(T);

        CHECK_CUDA(cudaMemcpyAsync(
            cached_graph->k_buf->memory(), k,
            prefix_bytes, cudaMemcpyDeviceToDevice, stream));

        CHECK_CUDA(cudaMemcpyAsync(
            cached_graph->v_buf->memory(), v,
            prefix_bytes, cudaMemcpyDeviceToDevice, stream));
    }

    // 尾部 [prefix_len, kv_capacity) 清零，避免读到未定义数据
    if (static_cast<size_t>(kv_capacity) > prefix_len) {
        const size_t tail_len   = static_cast<size_t>(kv_capacity) - prefix_len;
        const size_t tail_elems = tail_len *
                                  static_cast<size_t>(kv_num_heads) *
                                  static_cast<size_t>(hidden_dim);
        const size_t tail_bytes = tail_elems * sizeof(T);

        const size_t base_elems = static_cast<size_t>(prefix_len) *
                                  static_cast<size_t>(kv_num_heads) *
                                  static_cast<size_t>(hidden_dim);
        const size_t base_bytes = base_elems * sizeof(T);

        CHECK_CUDA(cudaMemsetAsync(
            reinterpret_cast<char *>(cached_graph->k_buf->memory()) + base_bytes,
            0, tail_bytes, stream));

        CHECK_CUDA(cudaMemsetAsync(
            reinterpret_cast<char *>(cached_graph->v_buf->memory()) + base_bytes,
            0, tail_bytes, stream));
    }

    //  Bias 构造：可见区 0，不可见区 -inf 
    // 可见区 [0, prefix_len) 置 0
    if (prefix_len > 0) {
        CHECK_CUDA(cudaMemsetAsync(
            cached_graph->bias_buf->memory(),
            0, prefix_len * sizeof(float), stream));
    }

    // 不可见区 [prefix_len, kv_capacity) 置 -inf
    // TODO：在这里支持回退prefix_len
    if (static_cast<size_t>(kv_capacity) > prefix_len) {
        const size_t neg_inf_count  = static_cast<size_t>(kv_capacity) - prefix_len;
        const size_t neg_inf_offset = prefix_len; // 元素偏移（以 float 计）

        dim3 block(256), grid(static_cast<unsigned>((neg_inf_count + 256 - 1) / 256));

        fill_neg_inf<<<grid, block, 0, stream>>>(
            reinterpret_cast<float*>(cached_graph->bias_buf->memory()) + neg_inf_offset,
            neg_inf_count);
    }

    // 执行 cuDNN FE 图 
    void* workspace_ptr = cached_graph->workspace_buf
                          ? cached_graph->workspace_buf->memory()
                          : nullptr;

    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensor_map;
    tensor_map.emplace(cached_graph->tQ,     const_cast<T*>(q));
    tensor_map.emplace(cached_graph->tK,     cached_graph->k_buf->memory());
    tensor_map.emplace(cached_graph->tV,     cached_graph->v_buf->memory());
    tensor_map.emplace(cached_graph->tO_gen, atten_val);
    tensor_map.emplace(cached_graph->tBias,  cached_graph->bias_buf->memory());

    auto st = cached_graph->graph.execute(cached_graph->handle, tensor_map, workspace_ptr);
    if (st.is_bad()) {
        throw std::runtime_error(std::string("[FE] execute failed: ") + st.get_message());
    }
}


} // llaisys::ops::nvidia::kernels

namespace llaisys::ops::nvidia::kernels {


template <typename T>
void atten3d_hdim128_decode_kernel(
    T*            out_atten,     // [seq_len, nhead, 128]
    const T*      q,             // [seq_len, nhead, 128]
    const T*      k,             // [total_len, nkvhead, 128]
    const T*      v,             // [total_len, nkvhead, 128]
    float         scale,
    size_t        seq_len,
    size_t        nhead,
    size_t        total_len,
    size_t        nkvhead,
    cudaStream_t  stream_in)
{
    // 固定参数（本算子场景）
    constexpr int64_t batch        = 1;
    constexpr int64_t hidden_dims  = 128;

    // 本次请求形状（不写入 cache，只用于本次判断/建图）
    const int64_t query_seq_len = static_cast<int64_t>(seq_len);   // 你的场景固定=1
    const int64_t num_query_heads = static_cast<int64_t>(nhead);
    const int64_t num_kv_heads    = static_cast<int64_t>(nkvhead);
    const int64_t kv_len          = static_cast<int64_t>(total_len);

    // 扩容策略
    constexpr int64_t INIT_KV_CAP   = 128;
    constexpr int64_t GROW_FACTOR   = 2;

    static std::mutex cache_mutex;
    static std::unique_ptr<CachedGraph> cached_graph;

    // 是否需要重建图
    bool need_rebuild = (!cached_graph) ||
                        (kv_len > cached_graph->cap_total_len) ||
                        (num_query_heads != cached_graph->cap_nhead) ||
                        (num_kv_heads   != cached_graph->cap_nkvhead);

    if (need_rebuild) {
        std::lock_guard<std::mutex> lock(cache_mutex);

        if ((!cached_graph) ||
            (kv_len > cached_graph->cap_total_len) ||
            (num_query_heads != cached_graph->cap_nhead) ||
            (num_kv_heads   != cached_graph->cap_nkvhead))
        {
            // 计算新的 KV capacity（向上扩容以减少后续重建频率）
            int64_t new_kv_capacity;
            if (!cached_graph) {
                new_kv_capacity = std::max<int64_t>(INIT_KV_CAP, kv_len);
            } else {
                new_kv_capacity = std::max<int64_t>(cached_graph->cap_total_len * GROW_FACTOR,
                                                    kv_len * GROW_FACTOR);
            }

            // 建图（按新容量）
            auto new_cached = create_graph<T>(
                batch,
                query_seq_len,
                num_query_heads,
                new_kv_capacity,   // total_len(capacity) for K/V/bias tensors in the graph
                num_kv_heads,
                hidden_dims,
                scale,
                stream_in);

            // 记录容量（create_graph 内部已分配 bias/k/v staging，并设置 cap_*）
            cached_graph = std::move(new_cached);

            std::cout << "=== Rebuild graph: KV capacity=" << new_kv_capacity
                      << ", nhead="   << num_query_heads
                      << ", nkvhead=" << num_kv_heads << " ===\n";
        }
    }

    // 执行（内部会做：拷贝 K/V 前缀、清零 K/V 尾部、bias 前缀=0 尾部=-inf、然后执行 graph）
    execute_graph<T>(cached_graph.get(),
                     out_atten, q, k, v,
                     static_cast<size_t>(kv_len),
                     stream_in);

}

// 显式实例化
template void atten3d_hdim128_decode_kernel<__half>(
    __half*, const __half*, const __half*, const __half*,
    float, std::size_t, std::size_t, std::size_t, std::size_t, cudaStream_t);

template void atten3d_hdim128_decode_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    float, std::size_t, std::size_t, std::size_t, std::size_t, cudaStream_t);

} // namespace llaisys::ops::nvidia::kernels