// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;        // 输入总元素个数
    uint32_t block_factor;  // 单核负责的元素个数（32字节对齐）
};
