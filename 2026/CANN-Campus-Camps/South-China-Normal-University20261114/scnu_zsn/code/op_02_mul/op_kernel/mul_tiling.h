// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength; // 数据总长度
    uint32_t tileNum;     // 切分的块数
};