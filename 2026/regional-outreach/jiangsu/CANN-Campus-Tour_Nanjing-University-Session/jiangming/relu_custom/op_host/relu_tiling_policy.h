#ifndef RELU_TILING_POLICY_H
#define RELU_TILING_POLICY_H

#include <stddef.h>
#include <stdint.h>

// This part is independent of CANN so the actual host policy can be tested on CPU.
namespace relu_custom {
constexpr uint32_t kElementsPerBlock = 16;  // 32 bytes / sizeof(float16)
constexpr uint32_t kBufferCount = 2;
constexpr uint64_t kUbReserve = 8192;
constexpr uint32_t kMaxTileLength = 8192;

struct TilingPlan {
    uint64_t totalLength;
    uint64_t blocksPerCore;
    uint32_t extraBlocks;
    uint32_t tileLength;
    uint32_t blockDim;
};

inline bool MakeTilingPlan(const int64_t* dims, size_t rank, uint32_t coreCount,
                          uint64_t ubBytes, TilingPlan& plan)
{
    if (dims == nullptr || rank != 4 || coreCount == 0 || ubBytes <= kUbReserve) {
        return false;
    }
    bool empty = false;
    for (size_t i = 0; i < rank; ++i) {
        if (dims[i] < 0) {
            return false;  // Runtime tiling requires resolved dimensions.
        }
        empty = empty || dims[i] == 0;
    }

    uint64_t total = empty ? 0 : 1;
    if (!empty) {
        for (size_t i = 0; i < rank; ++i) {
            // Keep the byte size within signed 64-bit address arithmetic.
            if (static_cast<uint64_t>(dims[i]) > (INT64_MAX / 2) / total) {
                return false;
            }
            total *= static_cast<uint64_t>(dims[i]);
        }
    }

    // Two input buffers and two output buffers, each holding half elements.
    uint64_t tile = (ubBytes - kUbReserve) / (2 * kBufferCount * 2);
    tile = tile / kElementsPerBlock * kElementsPerBlock;
    if (tile == 0) {
        return false;
    }
    tile = tile < kMaxTileLength ? tile : kMaxTileLength;

    const uint64_t blocks = (total + kElementsPerBlock - 1) / kElementsPerBlock;
    const uint32_t used = blocks == 0 ? 1 :
        static_cast<uint32_t>(blocks < coreCount ? blocks : coreCount);
    plan = {total, blocks / used, static_cast<uint32_t>(blocks % used),
            static_cast<uint32_t>(tile), used};
    return true;
}
}  // namespace relu_custom

#endif
