/*!
 * \file exp_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _EXP_TILING_DATA_H_
#define _EXP_TILING_DATA_H_

struct ExpTilingData {
    uint32_t totalLength; // 总元素数量
    uint32_t tileNum;     // 单核内分块数
};
#endif
