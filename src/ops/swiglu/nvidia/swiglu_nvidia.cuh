#include "llaisys.h"

#include <cstddef>
#include "../../../utils.hpp"

#include "../../../core/llaisys_core.hpp"



namespace llaisys::ops::nvidia {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, 
			size_t n, size_t d);

} // namespace llaisys::ops::nvidia

