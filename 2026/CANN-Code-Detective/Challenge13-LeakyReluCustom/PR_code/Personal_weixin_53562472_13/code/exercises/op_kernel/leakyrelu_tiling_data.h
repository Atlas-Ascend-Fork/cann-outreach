/*!
 * \file leakyrelu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _LEAKYRELU_TILING_DATA_H_
#define _LEAKYRELU_TILING_DATA_H_

struct LeakyreluTilingData {
    uint32_t totalLength; // 总元素数量
    uint32_t tileNum;     // 单核内分块数
    float negativeSlope;  // 属性：负半轴斜率 α（由 op_host Tiling 从 attr 读出后经 tilingData 下发）
};
#endif
