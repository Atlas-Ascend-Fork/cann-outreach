/*!
 * \file tanh_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _TANH_TILING_DATA_H_
#define _TANH_TILING_DATA_H_

struct TanhTilingData {
    uint32_t totalLength; // 总元素数量
    uint32_t tileNum;     // 单核内分块数
    uint32_t totalNum;
    uint32_t blockFactor;
    uint32_t ubFactor;
};
#endif
