#pragma once

#include <string>
#include <cuda_runtime.h>
#include <system_error>
#include <limits>
#include <type_traits>  

#include <cuda_fp16.h>          
#include <cuda_bf16.h>          

#include <cublas_v2.h>   
#include <cudnn.h>
#include <cudnn_frontend.h> 




namespace llaisys::device::nvidia::utils {

/*----------------CUDA----------------*/
inline const std::error_category &cudaErrorCategory() noexcept {
    // 单例
    static struct : std::error_category {
        char const *name() const noexcept override {
            return "cuda";
        }

        std::string message(int ev) const override {
            return cudaGetErrorString(static_cast<cudaError_t>(ev));
        }
    } category;

    return category;
}


inline std::error_code makeCudaErrorCode(cudaError_t e) noexcept {
    return std::error_code(static_cast<int>(e), cudaErrorCategory());
}


inline void throwCudaError(cudaError_t err, const char *file, int line) {
    throw std::system_error(makeCudaErrorCode(err),
                            std::string(file ? file : "??") + ":" + std::to_string(line));
}

/*----------------cuBLAS----------------*/
inline const std::error_category& cublasErrorCategory() noexcept {
    static struct : std::error_category {
        const char* name() const noexcept override { return "cublas"; }
        std::string message(int ev) const override {
            return cublasGetStatusString(static_cast<cublasStatus_t>(ev));
        }
    } category;
    return category;
}

inline std::error_code makeCublasErrorCode(cublasStatus_t s) noexcept {
    return std::error_code(static_cast<int>(s), cublasErrorCategory());
}

inline void throwCublasError(cublasStatus_t s, const char* file, int line) {
    throw std::system_error(makeCublasErrorCode(s),
                            std::string(file ? file : "??") + ":" + std::to_string(line));

}

/*----------------cuDNN----------------*/
inline const std::error_category& cudnnErrorCategory() noexcept {
    static struct : std::error_category {
        const char* name() const noexcept override { return "cudnn"; }
        std::string message(int ev) const override {
            return cudnnGetErrorString(static_cast<cudnnStatus_t>(ev));
        }
    } category;
    return category;
}

inline std::error_code makeCudnnErrorCode(cudnnStatus_t s) noexcept {
    return std::error_code(static_cast<int>(s), cudnnErrorCategory());
}

inline void throwCudnnError(cudnnStatus_t s, const char* file, int line) {
    throw std::system_error(makeCudnnErrorCode(s),
                            std::string(file ? file : "??") + ":" + std::to_string(line));
}


/*----------------helper--------------*/
template<typename T>
__device__ __host__ float load_as_float(const T* p) {
    if constexpr (std::is_same_v<T, __half>)        return __half2float(*p);
    if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(*p);
    return *p; // float
}

template<typename T>
__device__ __host__ inline void store_from_float(T* p, float v) {
    if constexpr (std::is_same_v<T, __half>)        { *p = __float2half(v); return; }
    if constexpr (std::is_same_v<T, __nv_bfloat16>) { *p = __float2bfloat16(v); return; }
    *p = v; // float
}


} // namespace llaisys::device::nvidia::utils


#define CHECK_CUDA(expr) \
    do { \
        cudaError_t err__ = (expr); \
        if (err__ != cudaSuccess)  { \
            ::llaisys::device::nvidia::utils::throwCudaError(err__, __FILE__, __LINE__); \
        } \
    } while (0)

#define CHECK_CUBLAS(expr) \
    do { \
        cublasStatus_t st__ = (expr); \
        if (st__ != CUBLAS_STATUS_SUCCESS) { \
            ::llaisys::device::nvidia::utils::throwCublasError(st__, __FILE__, __LINE__); \
        } \
    } while (0)

#define CHECK_CUDNN(expr) \
    do { \
        cudnnStatus_t st__ = (expr); \
        if (st__ != CUDNN_STATUS_SUCCESS) { \
            ::llaisys::device::nvidia::utils::throwCudnnError(st__, __FILE__, __LINE__); \
        } \
    } while(0)





inline unsigned int safe_grid_size(size_t n, unsigned int block_size) {
    if (block_size == 0) {
        throw std::invalid_argument("block_size must be > 0");
    }

    size_t grid_size = (n + block_size - 1) / block_size;

    if (grid_size > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("Grid size exceeds CUDA limit (unsigned int max)");
    }

    return static_cast<unsigned int>(grid_size);    
}

