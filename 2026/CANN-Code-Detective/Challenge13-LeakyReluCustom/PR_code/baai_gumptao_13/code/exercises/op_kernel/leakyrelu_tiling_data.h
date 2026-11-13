#ifndef _LEAKYRELU_TILING_DATA_H_
#define _LEAKYRELU_TILING_DATA_H_

#include <cstdint>

struct LeakyreluTilingData {
    int64_t totalNum;
    int64_t blockFactor;
    int64_t ubFactor;
    float negativeSlope;
};

#endif
