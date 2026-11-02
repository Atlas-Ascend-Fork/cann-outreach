/*!
 * \file exp_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _EXP_TILING_DATA_H_
#define _EXP_TILING_DATA_H_

struct ExpTilingData {
    int32_t totalNum;                // 元素总数量（所有维度乘积）
    int32_t coreNum;                 // 实际使用核数（= block 数上限）
    int32_t blockFactor;             // 每核基础元素数（512 元素对齐，>= 4KB/核）
    int32_t blockNum;                // 实际 block 数 = ceil(totalNum / blockFactor)
    int32_t ubFactor;                // 单次 UB tile 元素数（256B 对齐）
    int32_t ubLoopOfFormerBlock;     // 首 block 的 UB 循环次数
    int32_t ubTailOfFormerBlock;     // 首 block 尾 tile 元素数
    int32_t ubLoopOfTailBlock;       // 尾 block 的 UB 循环次数
    int32_t ubTailOfTailBlock;       // 尾 block 尾 tile 元素数
};
#endif
