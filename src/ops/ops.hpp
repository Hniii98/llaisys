#pragma once	

#include "add/op.hpp"
#include "argmax/op.hpp"
#include "embedding/op.hpp"
#include "linear/op.hpp"
#include "rearrange/op.hpp"
#include "rms_norm/op.hpp"
#include "rope/op.hpp"
#include "self_attention/op.hpp"
#include "swiglu/op.hpp"


#include <limits>


namespace llaisys::ops::nvidia {
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
}