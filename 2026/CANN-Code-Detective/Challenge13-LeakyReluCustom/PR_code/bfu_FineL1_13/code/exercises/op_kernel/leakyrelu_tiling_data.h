/*!
 * \file leakyrelu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _LEAKYRELU_TILING_DATA_H_
#define _LEAKYRELU_TILING_DATA_H_

#include <cstdint>

struct LeakyreluTilingData {
    int64_t totalNum = 0;       // 输入Tensor的总元素数
    int64_t blockFactor = 1;    // 每个核处理的元素数（按DataCopy对齐）
    int64_t ubFactor = 0;       // 每个核每次循环处理的元素数
    float negativeSlope = 0.01F; // LeakyReLU负半轴斜率
    uint32_t reserved = 0;      // 显式补齐到8字节，保证TilingData布局稳定
};

#endif
