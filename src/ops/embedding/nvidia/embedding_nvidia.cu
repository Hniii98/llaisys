#include "llaisys.h"
#include "../../ops.hpp"
#include "../../../device/nvidia/utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

template <typename T>
__global__ void embedding_kernel(T *out,
                                 const int64_t *index_list,
                                 const T *weight,
                                 size_t row_size,
                                 size_t list_length) {
    for (size_t row = blockIdx.x; row < list_length; row += gridDim.x) {
        size_t weight_index = static_cast<size_t>(index_list[row]);
        const T* weight_ptr = weight + (weight_index * row_size); 
        T *out_ptr = out + (row * row_size);

        for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
            out_ptr[i] = weight_ptr[i];
        }
    }
}


} // anomynous namespace

namespace llaisys::ops::nvidia {

void embedding(std::byte *out,
               const std::byte *index_list, size_t list_length,
               const std::byte *weight, size_t stride,
               llaisysDataType_t type) {
    int device_id = llaisys::core::context().runtime().deviceId();
    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    unsigned int grid_limit = prop.maxGridSize[0];
    unsigned int grid_size = std::min<unsigned int>(
        static_cast<unsigned int>(list_length),
        grid_limit);
    dim3 block(256), grid(grid_size);

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        embedding_kernel<float><<<grid, block, 0, s>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const int64_t*>(index_list),
            reinterpret_cast<const float*>(weight),
            stride,
            list_length);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        embedding_kernel<__half><<<grid, block, 0, s>>>(
            reinterpret_cast<__half*>(out),
            reinterpret_cast<const int64_t*>(index_list),
            reinterpret_cast<const __half*>(weight),
            stride,
            list_length);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        embedding_kernel<__nv_bfloat16><<<grid, block, 0, s>>>(
            reinterpret_cast<__nv_bfloat16*>(out),
            reinterpret_cast<const int64_t*>(index_list),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            stride,
            list_length);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}


} // llaisys::ops::nvidia