/*!
 * \file exp_tiling_data.h
 * \brief Shared host/kernel tiling data.
 */

#ifndef _EXP_TILING_DATA_H_
#define _EXP_TILING_DATA_H_

#include <cstdint>

struct ExpTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
};

#endif
