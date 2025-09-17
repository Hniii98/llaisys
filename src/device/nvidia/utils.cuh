#pragma once

#include <string>
#include <cuda_runtime.h>
#include <system_error>
#include <limits>



namespace llaisys::device::nvidia::utils {


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





} // namespace llaisys::device::nvidia::utils


#define CHECK_CUDA(expr) \
    do { \
        cudaError_t err__ = (expr); \
        if (err__ != cudaSuccess)  { \
            ::llaisys::device::nvidia::utils::throwCudaError(err__, __FILE__, __LINE__); \
        } \
    } while (0)


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

