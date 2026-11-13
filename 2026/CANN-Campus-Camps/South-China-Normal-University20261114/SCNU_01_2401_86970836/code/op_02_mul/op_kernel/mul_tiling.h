// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t blockLength; // 每核处理的元素数（host 按最终核数预计算好，核内免除法）
    uint32_t tileNum;     // 单核内循环块数
    uint32_t tileLength;  // 每块元素数
};