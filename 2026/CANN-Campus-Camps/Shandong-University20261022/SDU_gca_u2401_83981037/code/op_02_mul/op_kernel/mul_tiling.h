#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;      // 输入总元素数。
    uint32_t tileLength;  // 单个 tile 的容量，按 32 字节对齐。
};
