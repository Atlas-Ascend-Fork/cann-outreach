/*!
 * \file sigmoid_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SIGMOID_TILING_DATA_H_
#define _SIGMOID_TILING_DATA_H_

struct SigmoidTilingData {
    int64_t total_elems = 0;  // 总元素数量
    int64_t core_elems = 1;   // 每个核处理的元素数量
    int64_t tile_elems = 0;   // 每次 UB 循环处理的元素数量
};
#endif
