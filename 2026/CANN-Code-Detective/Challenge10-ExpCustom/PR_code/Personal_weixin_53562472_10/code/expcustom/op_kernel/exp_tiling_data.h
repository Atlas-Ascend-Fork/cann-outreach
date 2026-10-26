/*!
 * \file exp_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _EXP_TILING_DATA_H_
#define _EXP_TILING_DATA_H_

#include <cstdint>

struct ExpTilingData {
    int64_t totalNum = 0;    // 输入张量总元素数
    int64_t blockFactor = 1; // 单个 AIV 核处理的元素数（按 32B 对齐）
    int64_t ubFactor = 0;    // 核内每次 UB 循环处理的元素数
};
#endif
