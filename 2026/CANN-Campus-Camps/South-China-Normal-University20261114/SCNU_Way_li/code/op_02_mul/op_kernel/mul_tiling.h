#ifndef MUL_TILING_H
#define MUL_TILING_H

#include <cstdint>

// Host 与 Kernel 共用同一份结构体，length 的单位是元素。
struct MulTilingData {
    uint32_t length;
};

constexpr uint32_t MUL_TILE_LENGTH = 256;
constexpr int32_t MUL_BUFFER_NUM = 2;
constexpr uint32_t MUL_ALIGN_BYTES = 32;

#endif