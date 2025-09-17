#include "add_nvidia.cuh"
#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../device/nvidia/utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>


namespace {

template<typename T>
struct add_t {
  __device__ T operator()(T x, T y) const { return x + y; }
};

// TOOD: bf16 and fp16 vectorize
template<>
struct add_t<__half> {
  __device__ __half operator()(__half x, __half y) const { return __hadd(x, y); }
};

template<>
struct add_t<__nv_bfloat16> {
  __device__ __nv_bfloat16 operator()(__nv_bfloat16 x, __nv_bfloat16 y) const { return __hadd(x, y); }
};

template <typename T>
__global__ void add_kernel(T* c, const T* a, const T* b, size_t n) {
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) c[i] = add_t<T>()(a[i], b[i]); 
}

} // anonymous namespace


namespace llaisys::ops::nvidia {

void add(std::byte* c, const std::byte* a, const std::byte* b,
                llaisysDataType_t type, size_t n) {

  auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());


  dim3 block(256), grid(safe_grid_size(n, block.x)); // ceil division

  switch (type) {
    case LLAISYS_DTYPE_F32: {
      add_kernel<float><<<grid, block, 0, s>>>(
        reinterpret_cast<float*>(c),
        reinterpret_cast<const float*>(a),
        reinterpret_cast<const float*>(b),
        n);
      break;
    }
    case LLAISYS_DTYPE_F16: {
      add_kernel<__half><<<grid, block, 0, s>>>(
        reinterpret_cast<__half*>(c),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(b),
        n);
      break;
    }
    case LLAISYS_DTYPE_BF16: {
      add_kernel<__nv_bfloat16><<<grid, block, 0, s>>>(
        reinterpret_cast<__nv_bfloat16*>(c),
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        n);
      break;
    }
    default:
      EXCEPTION_UNSUPPORTED_DATATYPE(type); 
  }
}


} // namespace llaisys::ops::nvidia



