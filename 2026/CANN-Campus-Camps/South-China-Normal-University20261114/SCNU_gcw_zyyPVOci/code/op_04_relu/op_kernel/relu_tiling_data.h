
#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

#include <cstdint>

// ReLU 算子 Tiling 参数
struct ReluTilingData {
    uint32_t totalLength;  // 输入张量总元素数量
    uint32_t blockLength;  // 每个核处理的元素数量
    uint32_t tileLength;   // 每个 Tile 的元素数量
    uint32_t tileNum;      // 每个核的 Tile 数量参考值
};

#endif // _RELU_TILING_DATA_H_
