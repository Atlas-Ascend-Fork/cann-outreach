/*!
 * \file sigmoid_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SIGMOID_TILING_DATA_H_
#define _SIGMOID_TILING_DATA_H_

struct SigmoidTilingData {
    int64_t totalNum = 0;        // 总元素数量
    int64_t blockFactor = 1;     // 每个核标准处理的元素数量（尾核除外）
    int64_t lastBlockLength = 0; // 尾核实际处理的元素数量
    int64_t ubFactor = 0;        // 每次 UB 循环处理的元素数量
};
#endif
