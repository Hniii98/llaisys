#pragma once

#include "llaisys.h"
#include "../op.hpp"


namespace llaisys::ops::cpu {
	
void linear(std::byte *out, 
			const std::byte *in, 
			const std::byte *weight, 
			const std::byte *bias, 
			size_t sequence_length, 
			size_t embedding_dim, 
			size_t features_dim, 
			llaisysDataType_t type);
} // namespace llaisys::ops::cpu
