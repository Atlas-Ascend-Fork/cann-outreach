// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 一组为 128 个元素，两种精度的块边界均满足 32 字节对齐。
constexpr uint32_t MUL_GROUP_ELEMENTS = 128;

struct MulTilingData {
    uint32_t baseLength;  // 每核至少处理的元素数
    uint32_t extraCores;  // 前 extraCores 个核各多处理一组
};
