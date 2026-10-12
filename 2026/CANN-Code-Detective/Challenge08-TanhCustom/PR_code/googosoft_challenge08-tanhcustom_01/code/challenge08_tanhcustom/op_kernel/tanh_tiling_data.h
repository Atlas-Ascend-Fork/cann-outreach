/*!
 * \file tanh_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _TANH_TILING_DATA_H_
#define _TANH_TILING_DATA_H_

#include <cstdint>

// Kernel 内部分块及临时空间上限，不改变模板的 TilingData 布局。
constexpr int64_t TANH_MAX_TILE_LENGTH = 1024;
constexpr int64_t TANH_TMP_BUFFER_BYTES = 32 * 1024;

struct TanhTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 1;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
};
#endif
