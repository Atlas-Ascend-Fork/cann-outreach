/*!
 * \file leakyrelu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _LEAKYRELU_TILING_DATA_H_
#define _LEAKYRELU_TILING_DATA_H_

#include <cstdint>

// negative_slope 的默认值（与算子属性缺省值、样例取值保持一致）。
#define LEAKYRELU_DEFAULT_SLOPE 0.01f


#define LEAKYRELU_SLOPE_TAG 0x4C524C554C524C55LL // "LRLULRLU"

struct LeakyreluTilingData {
    int64_t totalNum = 0;                              // 总元素数量
    int64_t blockFactor = 1;                           // 每个核处理的元素数量（32B 对齐）
    int64_t ubFactor = 0;                              // 每次 UB 循环处理的元素数量（32B 对齐）
    int64_t slopeTag = 0;                              // 属性有效标记，见 LEAKYRELU_SLOPE_TAG
    float negativeSlope = LEAKYRELU_DEFAULT_SLOPE;     // negative_slope 属性，随 tiling 下发到 kernel
    float reserved = 0.0f;                             
};
#endif
