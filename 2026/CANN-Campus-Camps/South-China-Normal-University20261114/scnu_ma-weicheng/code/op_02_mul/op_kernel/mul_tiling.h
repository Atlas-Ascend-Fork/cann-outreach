// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;   // 元素总数
    uint32_t blockLength;   // 每核负责的元素个数（32B 对齐）
    uint32_t tileLength;    // 单次搬运/计算的元素个数（32B 对齐）
    uint32_t tileNum;       // 每核循环次数（向上取整）
};