// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 常量：double buffer 块数、单核内 tile 个数（host 与 kernel 共用，需保持一致）
constexpr uint32_t MUL_BUFFER_NUM = 2;
constexpr uint32_t MUL_TILE_NUM = 8;

struct MulTilingData {
    uint32_t length;      // 输入总元素数（8 * 2048 = 16384）
    uint32_t blockLength; // 每个核处理的元素数（32B 对齐）
    uint32_t tileNum;     // 每个核内的 tile 个数
    uint32_t tileLength;  // 每个 tile 的元素数（32B 对齐）
};
