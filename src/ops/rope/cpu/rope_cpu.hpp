#pragma once

#include "llaisys.h"


#include <cstddef>

namespace llaisys::ops::cpu {

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
		  llaisysDataType_t type, size_t sequence_length, size_t num_heads, size_t embedding_dim);
}