// Host and kernel share the same byte layout.
#pragma once
#include <cstdint>
struct MulTilingData {
    uint32_t length;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t tileNum;
};