/**
 * @file relu_op_tiling.cpp
 * @brief Host-side tiling computation for ReLU operator
 *
 * Tiling strategy for element-wise ReLU on 4D tensor [N4, N3, N2, N1]:
 * - Treat as flat 1D array of FP16 elements
 * - Split into blocks that fit in UB (need space for input + output + zero)
 * - Each block is 32-byte aligned for BlockDataCopy
 */

#include "relu_op_tiling.h"

#include <algorithm>
#include <cstdio>

namespace {
constexpr int32_t FP16_SIZE = 2;         // FP16: 2 bytes
constexpr int32_t COPY_ALIGN = 32;       // 32-byte alignment for DMA
constexpr int32_t DEFAULT_UB_SIZE = 192 * 1024;  // 192KB for Ascend 910
constexpr int32_t DEFAULT_CORE_NUM = 20; // 20 AI cores on Ascend 910
}  // namespace

void ReluTiling::CalcTiling(int32_t ubSizeBytes, int32_t coreNum)
{
    // 1. Calculate total elements
    totalElements = n4 * n3 * n2 * n1;

    if (totalElements <= 0) {
        blockCount = 0;
        blockSizeElements = 0;
        return;
    }

    // 2. Calculate max elements per block
    // UB is shared by all cores, but each core has its own UB partition.
    // For Ascend 910: total UB = 192KB, per core UB = 192KB / coreNum (approx)
    // Actually, each core has access to the full UB but they're partitioned.
    // For simplicity, we assume each core gets ubSizeBytes / coreNum.
    // However, for element-wise ops, we often use the full per-core UB.
    //
    // Conservative approach: each block needs space for:
    //   - localIn:  blockSize * 2 bytes
    //   - localOut: blockSize * 2 bytes
    //   - localZero: blockSize * 2 bytes
    // Total: blockSize * 6 bytes
    //
    // But in practice, we can reuse UB by processing in smaller chunks.
    // For simplicity and safety, let's use:
    //   blockSize * (2 + 2) = blockSize * 4 bytes (in + out, zero is fixed-size)
    //
    // Actually, for the simplest correct approach:
    // We allocate UB for one input block and one output block.
    // Zero can be a small fixed-size tensor (just one vector of zeros).
    //
    // Let's use: perCoreUB = ubSizeBytes (full UB per core in the kernel model)
    // blockSize * 2 (in) + blockSize * 2 (out) <= perCoreUB
    // blockSize <= perCoreUB / 4

    int32_t perCoreUb = ubSizeBytes;  // Each core has its own UB partition
    int32_t maxBlockSize = perCoreUb / (FP16_SIZE * 4);  // in + out = 4 bytes per element

    // Align down to 32-byte boundary (16 FP16 elements)
    int32_t alignElements = COPY_ALIGN / FP16_SIZE;  // 16 elements
    maxBlockSize = (maxBlockSize / alignElements) * alignElements;

    // Minimum block size: at least one 32-byte block
    if (maxBlockSize < alignElements) {
        maxBlockSize = alignElements;
    }

    // 3. Calculate block count
    blockSizeElements = maxBlockSize;
    blockCount = static_cast<int32_t>((totalElements + blockSizeElements - 1) / blockSizeElements);

    // 4. If block count is less than coreNum, we can make blocks larger
    //    to balance workload across cores (not critical for correctness)
    if (blockCount > 0 && blockCount < coreNum) {
        // Recalculate to use fewer, larger blocks
        int32_t newBlockCount = std::max(1, blockCount);
        blockSizeElements = static_cast<int32_t>(
            (totalElements + newBlockCount - 1) / newBlockCount);
        blockSizeElements = (blockSizeElements / alignElements) * alignElements;
        blockCount = static_cast<int32_t>(
            (totalElements + blockSizeElements - 1) / blockSizeElements);
    }

    printf("[ReLU Tiling] Shape: [%ld, %ld, %ld, %ld], Total elements: %ld\n",
           n4, n3, n2, n1, totalElements);
    printf("[ReLU Tiling] Block size: %d elements, Block count: %d, Cores: %d\n",
           blockSizeElements, blockCount, coreNum);
}
