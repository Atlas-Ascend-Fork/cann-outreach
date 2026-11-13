/*!
 * \file leakyrelu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _LEAKYRELU_TILING_DATA_H_
#define _LEAKYRELU_TILING_DATA_H_

struct LeakyreluTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 1;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
    float negativeSlope = 0.0f;  // 属性：负半轴斜率 α（由 op_host Tiling 从 attr 读出后经 tilingData 下发）
};
#endif
