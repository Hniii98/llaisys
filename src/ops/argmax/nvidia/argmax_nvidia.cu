#include "argmax_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../ops.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cub/cub.cuh>

namespace {

// TODO: support data in vals occurs to be NAN
template <typename T>
void argmax_cub(int64_t *max_idx, T *max_val,
                const T *vals, int64_t numel,
                cudaStream_t stream) {
  
  size_t temp_bytes = 0;

  // step 1. caculate temp storage size
  cub::DeviceReduce::ArgMax(nullptr, temp_bytes,
                            vals,
                            max_val, max_idx,
                            numel,
                            stream);

  auto device_storage = llaisys::core::context().runtime().allocateDeviceStorage(temp_bytes);
  void *temp_ptr = static_cast<void *>(device_storage->memory());

  // step 2. do the real argmax operation
  cub::DeviceReduce::ArgMax(temp_ptr, temp_bytes,
                            vals,
                            max_val, max_idx,
                            numel,
                            stream);
}

	
} // anonymous namespace


namespace llaisys::ops::nvidia {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
			llaisysDataType_t type, size_t numel) {
	
	auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
  
  switch (type) {
    case LLAISYS_DTYPE_F32: {
      argmax_cub<float>(
        reinterpret_cast<int64_t*>(max_idx),
        reinterpret_cast<float*>(max_val),
        reinterpret_cast<const float*>(vals),
        static_cast<int64_t>(numel),
        s);
      break;
    }
    case LLAISYS_DTYPE_F16: {
      argmax_cub<__half>(
        reinterpret_cast<int64_t*>(max_idx),
        reinterpret_cast<__half*>(max_val),
        reinterpret_cast<const __half*>(vals),
        static_cast<int64_t>(numel),
        s);
      break;
    }
    case LLAISYS_DTYPE_BF16: {
      argmax_cub<__nv_bfloat16>(
        reinterpret_cast<int64_t*>(max_idx),
        reinterpret_cast<__nv_bfloat16*>(max_val),
        reinterpret_cast<const __nv_bfloat16*>(vals),
        static_cast<int64_t>(numel),
        s);
      break;
    }
    default:
      EXCEPTION_UNSUPPORTED_DATATYPE(type); 
  }
}

} // namespace llaisys::ops::nvidia
