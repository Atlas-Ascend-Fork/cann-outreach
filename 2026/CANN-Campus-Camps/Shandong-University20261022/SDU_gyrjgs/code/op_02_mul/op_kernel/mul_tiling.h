
#pragma once

#include <cstdint>

// Mul 算子的 tiling 参数
struct MulTilingData {
    uint32_t length;      // 输入张量的元素总数
    uint32_t tileLength;  // 每个数据块的元素数
};
