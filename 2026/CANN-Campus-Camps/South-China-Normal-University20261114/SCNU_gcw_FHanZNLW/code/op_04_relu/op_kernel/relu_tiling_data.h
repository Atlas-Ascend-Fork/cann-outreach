/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

#include <cstdint>

struct ReluTilingData {
    uint32_t totalNum;    // 总元素数
    uint32_t blockFactor; // 每核处理元素数（32 元素对齐）
    uint32_t ubFactor;    // 每核内单次 UB 循环处理的元素数
};
#endif