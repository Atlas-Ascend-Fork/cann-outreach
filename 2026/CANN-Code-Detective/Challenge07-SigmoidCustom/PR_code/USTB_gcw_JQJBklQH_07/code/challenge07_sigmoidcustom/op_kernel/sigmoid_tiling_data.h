/*!
 * \file sigmoid_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SIGMOID_TILING_DATA_H_
#define _SIGMOID_TILING_DATA_H_

struct SigmoidTilingData {
    uint32_t totalLength; // 输入总元素数量
    uint32_t tileNum;     // 单核内分块数
};
#endif
