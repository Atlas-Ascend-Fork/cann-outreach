// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;  // 所有元素的总个数
    uint32_t tileNum;      // 每个核内的分块数
};
