/*!
 * \file abs_tiling_data.h
 * \brief Abs 算子 tiling 数据结构
 */

#ifndef _ABS_TILING_DATA_H_
#define _ABS_TILING_DATA_H_

struct AbsTilingData {
    int64_t totalNum = 0;    // 整网张量元素总数
    int64_t blockFactor = 0; // 单核连续负责的元素数（尾核按剩余量截断）
    int64_t tileNum = 0;     // 核内分块数，单块长度 = blockFactor / tileNum
};
#endif
