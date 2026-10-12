/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

#include <cstdint>

struct ReluTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 1;  // 每个核处理的基准元素数量
    int64_t ubFactor = 0;     // 单次搬入 UB 并参与计算的元素数量
};
#endif
