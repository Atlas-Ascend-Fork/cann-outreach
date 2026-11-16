#ifndef CUSTOM_RELU_TILING_DATA_H
#define CUSTOM_RELU_TILING_DATA_H

#include <cstdint>

struct ReluTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
};

#endif