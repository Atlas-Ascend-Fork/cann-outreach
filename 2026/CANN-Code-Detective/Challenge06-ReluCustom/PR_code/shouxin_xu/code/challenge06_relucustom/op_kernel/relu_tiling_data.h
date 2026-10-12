/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    uint32_t totalLength; // 输入张量的总元素数
    uint32_t tileNum;     // 每个核的数据分块数
};
#endif

