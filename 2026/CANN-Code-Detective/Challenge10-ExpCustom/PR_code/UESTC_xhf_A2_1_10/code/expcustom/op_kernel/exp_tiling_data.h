/*!
 * \file exp_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _EXP_TILING_DATA_H_
#define _EXP_TILING_DATA_H_

struct ExpTilingData {
    int64_t totalNum = 0;     // 输入张量的总元素数
    int64_t blockFactor = 1;  // 每个 AIV 核处理的元素数
    int64_t ubFactor = 0;     // 每次搬入 UB 的元素数
};
#endif
