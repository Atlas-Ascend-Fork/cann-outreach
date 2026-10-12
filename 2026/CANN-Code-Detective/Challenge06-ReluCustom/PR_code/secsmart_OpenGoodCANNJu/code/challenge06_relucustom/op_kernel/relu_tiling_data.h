/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    int64_t totalNum;    // 输入总元素数
    int64_t blockFactor; // 单核处理的元素数（多核切分）
    int64_t ubFactor;    // 单次搬入 UB 的元素数（核内分片）
};
#endif