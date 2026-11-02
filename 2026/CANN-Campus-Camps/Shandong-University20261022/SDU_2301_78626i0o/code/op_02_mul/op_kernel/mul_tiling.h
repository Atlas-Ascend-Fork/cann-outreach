// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;  // 输入数据的总元素数
    uint32_t blockLength;  // 每个核分配的元素数上限
    uint32_t tileLength;   // 每次循环处理的元素数上限
};