#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, 
		 	  size_t sequence_length, size_t embedding_dim, float eps);
}