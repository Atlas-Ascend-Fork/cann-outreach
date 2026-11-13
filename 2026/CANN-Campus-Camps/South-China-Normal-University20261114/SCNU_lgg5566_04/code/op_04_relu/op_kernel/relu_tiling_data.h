/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

#include <cstdint>

struct ReluTilingData {
    uint32_t totalLength; // 总元素数量
    uint32_t blockLength; // 单核基础元素数，按 512 元素对齐
    uint32_t tileLength;  // 单次搬运/计算元素数，32B 对齐
};
#endif
