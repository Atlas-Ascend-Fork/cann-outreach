// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;          // 总元素个数
    uint32_t perCoreLength;   // 每个核负责的元素个数
    uint32_t tileLength;      // 每次搬入 UB 的元素个数
    uint32_t tileNum;         // 每个核的循环次数
};