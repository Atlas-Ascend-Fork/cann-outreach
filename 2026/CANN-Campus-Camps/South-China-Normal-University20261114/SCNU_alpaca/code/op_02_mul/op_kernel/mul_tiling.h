// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength; // 待处理元素总数，例如 8 * 2048 = 16384
    uint32_t tileNum;     // 单核内分块（Tile）数量
};