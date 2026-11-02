/*!
 * \file sqrt_tiling_data.h
 * \brief Sqrt 算子 tiling 数据结构
 */

#ifndef _SQRT_TILING_DATA_H_
#define _SQRT_TILING_DATA_H_

struct SqrtTilingData {
    int64_t totalNum = 0;    // 整网张量元素总数
    int64_t blockFactor = 0; // 单核负责的元素数（尾核按剩余元素截断）
    int64_t ubFactor = 0;    // 单次搬入 UB 的元素数
};

#endif // _SQRT_TILING_DATA_H_