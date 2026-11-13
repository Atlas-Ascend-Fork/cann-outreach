/*!
 * \file abs_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _ABS_TILING_DATA_H_
#define _ABS_TILING_DATA_H_

struct AbsTilingData {
    uint32_t totalLength;  // 总元素数量
    uint32_t alignNum;     // 32B 对齐的元素数（fp32:8 / fp16:16）
    uint32_t blockFactor;  // 小核处理的元素数（32B 对齐）
    uint32_t tailBlockNum; // 大核个数（比小核多处理一个 32B 块）
};
#endif
