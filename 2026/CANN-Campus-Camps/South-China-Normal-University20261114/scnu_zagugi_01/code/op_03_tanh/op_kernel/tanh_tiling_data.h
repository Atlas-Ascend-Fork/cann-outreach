#ifndef _TANH_TILING_DATA_H_
#define _TANH_TILING_DATA_H_

#include <cstdint>

struct TanhTilingData {
    uint32_t totalLength;  // 输入张量总元素数
    uint32_t tileNum;      // 每个核的分块数量
};

#endif