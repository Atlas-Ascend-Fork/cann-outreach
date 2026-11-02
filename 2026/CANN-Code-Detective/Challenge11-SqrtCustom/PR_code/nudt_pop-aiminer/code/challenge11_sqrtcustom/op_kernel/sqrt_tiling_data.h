#ifndef SQRT_CUSTOM_TILING_DATA_H
#define SQRT_CUSTOM_TILING_DATA_H

#include <cstdint>

// Shared, fixed-width ABI; retain the field names used by the supplied UT.
struct SqrtTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
};
static_assert(sizeof(SqrtTilingData) == 24, "Unexpected Sqrt tiling layout");

#endif
