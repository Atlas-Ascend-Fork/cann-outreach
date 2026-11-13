// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;  // 总元素个数
    uint32_t tileNum;      // 切分块数 (对应 Host 侧的 TILE_NUM)
};