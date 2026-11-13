#pragma once

#include <cstdint>

constexpr uint32_t MUL_DATA_BLOCK_BYTES = 32;

struct MulTilingData {
    uint32_t length;
    uint32_t tileLength;
};