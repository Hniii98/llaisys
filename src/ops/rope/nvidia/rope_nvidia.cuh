#pragma once 

#include "llaisys.h"

#include <cstddef>
#include "../../../utils.hpp"
#include "../../../core/llaisys_core.hpp"

namespace llaisys::ops::nvidia {

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theat, 
		  llaisysDataType_t type, size_t sequence_length, size_t num_heads, size_t embedding_dim);
}