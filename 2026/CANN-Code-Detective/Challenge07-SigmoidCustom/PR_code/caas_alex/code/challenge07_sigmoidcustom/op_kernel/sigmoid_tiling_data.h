/*!
 * \file sigmoid_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SIGMOID_TILING_DATA_H_
#define _SIGMOID_TILING_DATA_H_

#include <cstdint>

struct SigmoidTilingData {
    int64_t totalLength = 0;
    int64_t blockLength = 0;
    int64_t tileNum = 0;
    int64_t tileLength = 0;
};
#endif
