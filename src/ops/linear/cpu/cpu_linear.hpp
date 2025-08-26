#pragma once

#include "llaisys.h"
#include "../op.hpp"


namespace llaisys::ops::cpu {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
} // namespace llaisys::ops::cpu
