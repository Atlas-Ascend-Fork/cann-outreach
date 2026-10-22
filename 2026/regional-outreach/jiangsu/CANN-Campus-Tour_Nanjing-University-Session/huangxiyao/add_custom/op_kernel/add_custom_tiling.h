/**
 * \file add_custom_tiling.h
 * \brief add_custom 算子的 Tiling 数据结构，Host 侧（TilingFunc）与 Kernel 侧（核函数）共享。
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

#include <cstdint>

struct AddCustomTilingData {
    uint32_t totalLength;  // 输入张量总元素数（N2 * N1）
    uint32_t blockDim;     // 参与计算的 AICore 核数
};

#endif  // ADD_CUSTOM_TILING_H
