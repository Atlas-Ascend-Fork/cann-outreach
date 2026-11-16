// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;       // 输入/输出张量的元素总个数（如 8 * 2048 = 16384）
    uint32_t blockLength;  // 每个 AI Core 负责的元素个数（已按 32B 对齐）
    uint32_t tileLength;   // 单核内每个 tile 的元素个数（已按 32B 对齐）
};