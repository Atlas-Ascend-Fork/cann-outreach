#pragma once

#include <cstdint>

// Host 和 Kernel 共用的分块参数
struct MulTilingData {
    uint32_t totalLength;  // 总元素个数
    uint32_t tileNum;      // 单核分块参数
};