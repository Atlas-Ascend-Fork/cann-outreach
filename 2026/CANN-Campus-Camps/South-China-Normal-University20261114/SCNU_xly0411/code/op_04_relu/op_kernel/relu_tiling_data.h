/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    uint32_t totalNum;    // 总元素数量
    uint32_t blockFactor; // 每核处理元素数
    uint32_t ubFactor;    // 单次 UB 循环元素数
    uint32_t reserved;    // 保留字段（对齐 16 字节）
};
#endif
