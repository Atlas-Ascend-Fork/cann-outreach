/*!
 * \file exp_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _EXP_TILING_DATA_H_
#define _EXP_TILING_DATA_H_

#include <cstdint>

struct ExpTilingData {
    uint32_t totalLength; // 输入总元素数
    uint32_t tileNum;     // 每核分块数，实际循环次数为 tileNum * BUFFER_NUM
};
#endif
