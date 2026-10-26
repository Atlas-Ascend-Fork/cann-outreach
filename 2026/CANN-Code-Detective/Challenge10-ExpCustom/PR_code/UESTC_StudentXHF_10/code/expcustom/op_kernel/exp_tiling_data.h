/*!
 * \file exp_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _EXP_TILING_DATA_H_
#define _EXP_TILING_DATA_H_

#include <cstdint>

struct ExpTilingData {
    uint64_t totalNum;
    uint64_t blockFactor;
    uint64_t ubFactor;
};
#endif
