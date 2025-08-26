#pragma once 

#include "llaisys.h"

#include <vector>

#include <cstddef>

namespace llaisys::ops::cpu {
	void rearrange(	std::byte *out,
					std::byte *in, 
					const std::vector<size_t> &in_shape,
					const std::vector<ptrdiff_t> &in_stride,
					llaisysDataType_t in_dtype);
	}


