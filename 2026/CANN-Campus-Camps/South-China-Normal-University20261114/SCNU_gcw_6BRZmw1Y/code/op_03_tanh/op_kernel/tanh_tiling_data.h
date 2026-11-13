/*!
 * \file tanh_tiling_data.h
 * \brief Tanh 算子 Tiling 数据结构
 */

#ifndef _TANH_TILING_DATA_H_
#define _TANH_TILING_DATA_H_

#include <cstdint>

struct TanhTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t tmpBufferSize;
};

#endif