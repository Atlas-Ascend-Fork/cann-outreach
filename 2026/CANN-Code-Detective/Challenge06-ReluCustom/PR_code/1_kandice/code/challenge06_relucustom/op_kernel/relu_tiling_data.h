/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

#include <cstdint>

struct ReluTilingData {
    int64_t totalLength = 0;
    int64_t blockLength = 0;
    int64_t tileNum = 0;
    int64_t tileLength = 0;
};
#endif
