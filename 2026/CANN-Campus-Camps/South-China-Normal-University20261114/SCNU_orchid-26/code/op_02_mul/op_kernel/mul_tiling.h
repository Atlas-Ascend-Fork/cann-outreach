// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;
    uint32_t tileNum;   // 单核内切分块数（与 Kernel 侧 TQue 循环配合）
};