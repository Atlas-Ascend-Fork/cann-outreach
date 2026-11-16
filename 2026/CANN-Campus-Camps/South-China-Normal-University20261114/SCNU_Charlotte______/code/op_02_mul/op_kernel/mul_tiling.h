// Tiling结构体定义的头文件
#pragma once

#include <cstdint>


struct MulTilingData {
    uint32_t totalLength; // 待处理数据总长度（元素个数），形状 (8, 2048) 时为 16384
    uint32_t tileNum;     // 单核内将数据进一步切分的块数（Tile），例如 8
};

