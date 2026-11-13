// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;  // 总元素个数，例如 8 * 2048 = 16384
    uint32_t tileNum;   
};