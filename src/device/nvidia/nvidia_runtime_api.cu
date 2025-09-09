#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "utils.cuh"

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    return device_count;
}

void setDevice(int device_id) {
    CHECK_CUDA(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    CHECK_CUDA(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t cuda_stream;
    CHECK_CUDA(cudaStreamCreate(&cuda_stream));
    return static_cast<llaisysStream_t>(cuda_stream);
}

void destroyStream(llaisysStream_t stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    CHECK_CUDA(cudaStreamDestroy(cuda_stream)); 
}
void streamSynchronize(llaisysStream_t stream) {
    CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if(ptr) {
        CHECK_CUDA(cudaFree(ptr));
    }
}

// 分配锁页内存
void* mallocHost(size_t size) { 
    void* ptr = nullptr;
        CHECK_CUDA(cudaMallocHost(&ptr, size));
    return ptr;
}


void freeHost(void *ptr) {
    if(ptr) {       
        CHECK_CUDA(cudaFreeHost(ptr));
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CHECK_CUDA(cudaMemcpy(dst,
                          src,
                          size,
                          static_cast<cudaMemcpyKind>(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CHECK_CUDA(cudaMemcpyAsync(dst,
                               src,
                               size,
                               static_cast<cudaMemcpyKind>(kind),
                               static_cast<cudaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
