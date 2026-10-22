/**
 * @file add_custom_tiling.h
 * @brief add_custom 算子 Host / Kernel 两侧共享的 Tiling 数据结构。
 *
 * 结构体必须使用标准 C++ 定义，且字段类型、数量、顺序在 Host 侧与 Kernel 侧必须完全一致，
 * 二者通过二进制内存直接拷贝通信（无序列化头），因此两侧都要 include 本头文件。
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

#include <cstdint>

struct AddCustomTilingData {
    uint32_t totalLength;  // 本次计算的总元素数 (N2 * N1)
    uint32_t tileNum;      // 单核数据切分出的 tile 数量
};

#endif  // ADD_CUSTOM_TILING_H
