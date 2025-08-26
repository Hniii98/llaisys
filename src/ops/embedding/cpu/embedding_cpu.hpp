#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *dst, const std::byte *src, llaisysDataType_t type, size_t numel);
}






