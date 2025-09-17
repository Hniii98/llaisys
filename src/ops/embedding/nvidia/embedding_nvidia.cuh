#pragma once 
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void embedding(std::byte *out,
			   const std::byte *index_list,
			   size_t list_length,
			   const std::byte *weight,
			   size_t stride,
			   llaisysDataType_t type);
}