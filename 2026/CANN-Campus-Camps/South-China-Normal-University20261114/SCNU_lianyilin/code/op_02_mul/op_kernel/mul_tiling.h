// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength; // 输入张量的元素总数
    uint32_t tileNum;     // 每个核的分块数
};