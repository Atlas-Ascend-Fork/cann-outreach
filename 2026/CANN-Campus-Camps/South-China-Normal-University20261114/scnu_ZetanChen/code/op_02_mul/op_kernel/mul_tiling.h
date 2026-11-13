#ifndef MUL_TILING_H
#define MUL_TILING_H

#include <cstdint>

struct MulTilingData {
    uint32_t length;      // 总元素数
    uint32_t tileLength;  // 单次处理的最大元素数
};

#endif