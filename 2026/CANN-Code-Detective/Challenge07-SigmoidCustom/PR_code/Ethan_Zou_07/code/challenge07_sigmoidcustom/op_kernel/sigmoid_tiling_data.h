/*!
 * \file sigmoid_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SIGMOID_TILING_DATA_H_
#define _SIGMOID_TILING_DATA_H_

#include <cstdint>

struct SigmoidTilingData {
    int64_t totalNum;
    int64_t blockFactor;
    int64_t ubFactor;
};
#endif
