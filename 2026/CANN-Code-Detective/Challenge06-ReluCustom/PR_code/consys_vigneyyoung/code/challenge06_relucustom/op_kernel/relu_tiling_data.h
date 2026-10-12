/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    uint32_t totalNum;    // 总元素数量
    uint32_t blockFactor; // 每核处理的元素数量
    uint32_t ubFactor;    // 单次 UB 计算的元素数量
};
#endif
