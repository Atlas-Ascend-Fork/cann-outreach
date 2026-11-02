#pragma once
#include <cstdint>

struct MulTilingData {
    uint32_t length;      // Total number of elements.
    uint32_t tileLength;  // Number of elements in each tile.
    uint32_t tileCount;   // Total number of tiles.
};