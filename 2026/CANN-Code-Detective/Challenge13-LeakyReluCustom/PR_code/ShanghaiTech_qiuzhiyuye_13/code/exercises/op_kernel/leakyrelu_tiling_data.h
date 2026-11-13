/*!
 * \file leakyrelu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _LEAKYRELU_TILING_DATA_H_
#define _LEAKYRELU_TILING_DATA_H_

#include <cstdint>

struct LeakyreluTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 1;
    int64_t ubFactor = 0;
    float negativeSlope = 0.01f;
};
#endif
