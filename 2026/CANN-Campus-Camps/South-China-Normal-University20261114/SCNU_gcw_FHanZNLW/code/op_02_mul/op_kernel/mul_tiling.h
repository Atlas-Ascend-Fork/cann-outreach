// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;       // 元素总个数
    uint32_t blockLength;  // 每核元素个数（32B 对齐，最后一核可能更少）
    uint32_t tileLength;   // 每块元素个数（32B 对齐）
};