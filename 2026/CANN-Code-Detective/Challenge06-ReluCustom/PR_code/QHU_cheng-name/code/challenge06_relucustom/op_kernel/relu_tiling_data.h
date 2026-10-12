/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    int64_t totalNum;    // 总元素数量
    int64_t blockFactor; // 每核处理元素数（多核切分后）
    int64_t ubFactor;    // UB 单块元素数（每核分块后的单块大小）
};

#endif
