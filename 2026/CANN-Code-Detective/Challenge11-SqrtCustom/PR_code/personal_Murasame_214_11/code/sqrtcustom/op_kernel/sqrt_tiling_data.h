/*!
 * \file sqrt_tiling_data.h
 * \brief Sqrt 算子 tiling 数据结构
 */

#ifndef _SQRT_TILING_DATA_H_
#define _SQRT_TILING_DATA_H_

struct SqrtTilingData {
    int64_t totalNum = 0;    // 整网张量元素总数
    int64_t blockFactor = 0; // 每核按 tile 轮转处理到的元素数上限
    int64_t ubFactor = 0;    // 单个 tile 的元素数（32B 对齐）
};

#endif // _SQRT_TILING_DATA_H_