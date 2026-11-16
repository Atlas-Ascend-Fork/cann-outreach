// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;   // 数据总长度
    uint32_t tileNum;       // 每核分块数
    uint32_t blockLength;   // 每核处理的数据长度
    uint32_t tileLength;    // 每块处理的数据长度
};