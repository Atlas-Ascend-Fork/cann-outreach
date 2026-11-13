#ifndef COURSE_RELU_TILING_DATA_H
#define COURSE_RELU_TILING_DATA_H
#include <cstdint>

// Streaming tensors use two input/output buffers.
#define RELU_BUFFER_NUM 2

struct ReluTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t blockLength;
    uint32_t tileLength;
};
#endif
