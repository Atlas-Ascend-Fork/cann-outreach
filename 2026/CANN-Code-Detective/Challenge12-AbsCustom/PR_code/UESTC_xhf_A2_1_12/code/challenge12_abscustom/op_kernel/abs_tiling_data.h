/*!
 * \file abs_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _ABS_TILING_DATA_H_
#define _ABS_TILING_DATA_H_

#include <cstdint>

struct AbsTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
};
#endif
