/*!
 * \file sigmoid_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SIGMOID_TILING_DATA_H_
#define _SIGMOID_TILING_DATA_H_

#include <cstdint>

// Host 与 kernel 共享的切分参数，单位均为元素个数。
struct SigmoidTilingData {
    int64_t totalNum = 0;     // 输入张量的总元素数
    int64_t blockFactor = 1;  // 每个 block 分配的元素数
    int64_t ubFactor = 0;     // 每次放入 UB 的元素数
};
#endif
