/*!
 * \file exp_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _EXP_TILING_DATA_H_
#define _EXP_TILING_DATA_H_

#include <cstdint>

struct ExpTilingData {
    int64_t totalNum = 0;    // 输入总元素数量
    int64_t blockFactor = 1; // 每个核处理的元素数量
    int64_t ubFactor = 0;    // 单次 UB 循环处理的元素数量
};
#endif
