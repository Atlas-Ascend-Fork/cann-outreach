/*!
 * \file sigmoid_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SIGMOID_TILING_DATA_H_
#define _SIGMOID_TILING_DATA_H_

struct SigmoidTilingData {
    uint32_t totalLength; // 输入张量的总元素数
    uint32_t tileNum;     // 每个核内的数据分块数
};
#endif

