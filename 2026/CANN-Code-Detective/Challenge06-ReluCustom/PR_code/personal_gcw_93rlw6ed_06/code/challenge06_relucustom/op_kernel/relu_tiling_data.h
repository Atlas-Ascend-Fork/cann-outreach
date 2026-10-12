/*!
 * ile relu_tiling_data.h
 * rief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    uint32_t totalNum;        // 总元素数量
    uint32_t blockFactor;     // 每核处理的元素数（前 N-1 核）
    uint32_t lastBlockLength; // 最后一个核实际处理的元素数（尾块）
    uint32_t ubFactor;        // 单次搬运/计算的元素数
};
#endif
