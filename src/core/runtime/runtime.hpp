#pragma once
#include "../core.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/allocator.hpp"


#if defined(ENABLE_NVIDIA_API)
#include "../../device/nvidia/utils.cuh"
#include <cublas_v2.h>   // cuBLAS handle only when NVIDIA API is enabled
//#include "../../device/nvidia/utils.cuh"
#endif


namespace llaisys::core {
class Runtime {
private:
    llaisysDeviceType_t _device_type;
    int _device_id;
    const LlaisysRuntimeAPI *_api;
    MemoryAllocator *_allocator;
    bool _is_active;
    void _activate();
    void _deactivate();
    llaisysStream_t _stream;
    Runtime(llaisysDeviceType_t device_type, int device_id);

#if defined(ENABLE_NVIDIA_API)
    cublasHandle_t _cublas_handle;   // cuBLAS handle (only valid if ENABLE_NVIDIA_API)
#endif

public:
    friend class Context;

    ~Runtime();

    // Prevent copying
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    // Prevent moving
    Runtime(Runtime &&) = delete;
    Runtime &operator=(Runtime &&) = delete;

    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    bool isActive() const;

    const LlaisysRuntimeAPI *api() const;

    storage_t allocateDeviceStorage(size_t size);
    storage_t allocateHostStorage(size_t size);
    void freeStorage(Storage *storage);

    llaisysStream_t stream() const;
#if defined(ENABLE_NVIDIA_API)
    cublasHandle_t cublasHandle() const;
#endif
    void synchronize() const;
};
} // namespace llaisys::core
