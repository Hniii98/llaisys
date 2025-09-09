#pragma once

#include <string>
#include <cuda_runtime.h>
#include <system_error>


namespace llaisys::device::nvidia::utils {

// 单例
inline std::error_category const &cudaErrorCategory() noexcept {
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


inline void throwCudaError(cudaError_t err, char const *file, int line) {
    throw std::system_error(makeCudaErrorCode(err),
                            std::string(file ? file : "??") + ":" + std::to_string(line));
}





} // 


#define CHECK_CUDA(expr) \
    do { \
        cudaError_t err__ = (expr); \
        if (err__ != cudaSuccess) [[unlikely]] { \
            ::llaisys::device::nvidia::utils::throwCudaError(err__, __FILE__, __LINE__); \
        } \
    } while (0)