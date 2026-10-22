/**
 * @file inplace_rsqrt_tiling.cpp
 * @brief Host-side tiling computation for InplaceRsqrt operator
 *
 * Tiling strategy (in-place, single buffer):
 * - Total UB per core: UB_SIZE (192KB for 910)
 * - In-place: 1 buffer of FP16 data
 * - Max block = UB_SIZE / 2 bytes per element = 98304 elements (for 910)
 * - Aligned down to 32-byte boundary (16 FP16 elements)
 */

#include "inplace_rsqrt_tiling.h"

#include <algorithm>
#include <cstdio>

namespace {
constexpr int32_t FP16_SIZE = 2;              // FP16: 2 bytes
constexpr int32_t COPY_ALIGN = 32;            // 32-byte DMA alignment
constexpr int32_t DEFAULT_UB_SIZE = 192 * 1024;   // 192KB (Ascend 910)
constexpr int32_t DEFAULT_CORE_NUM = 20;         // 20 AI cores
}  // namespace

void RsqrtTiling::CalcTiling(int32_t ubSizeBytes, int32_t coreNum)
{
    // 1. Total elements from 4D shape
    totalElements = n4 * n3 * n2 * n1;

    if (totalElements <= 0) {
        blockCount = 0;
        blockSizeElements = 0;
        return;
    }

    // 2. Max elements per block (in-place: single buffer)
    //    block * FP16_SIZE <= ubSizeBytes
    //    block <= ubSizeBytes / 2
    int32_t maxBlockSize = ubSizeBytes / FP16_SIZE;

    // Align down to 32-byte boundary (16 FP16 elements)
    int32_t alignElements = COPY_ALIGN / FP16_SIZE;  // 16
    maxBlockSize = (maxBlockSize / alignElements) * alignElements;

    if (maxBlockSize < alignElements) {
        maxBlockSize = alignElements;
    }

    blockSizeElements = maxBlockSize;

    // 3. Block count
    blockCount = static_cast<int32_t>(
        (totalElements + blockSizeElements - 1) / blockSizeElements);

    // 4. Balance across cores: if fewer blocks than cores,
    //    enlarge blocks so each core gets at most 1 block
    if (blockCount > 0 && blockCount < coreNum) {
        int32_t targetBlocks = std::max(1, blockCount);
        blockSizeElements = static_cast<int32_t>(
            (totalElements + targetBlocks - 1) / targetBlocks);
        blockSizeElements = (blockSizeElements / alignElements) * alignElements;
        blockCount = static_cast<int32_t>(
            (totalElements + blockSizeElements - 1) / blockSizeElements);
    }

    printf("[InplaceRsqrt Tiling] Shape: [%ld, %ld, %ld, %ld], Total: %ld elements\n",
           n4, n3, n2, n1, totalElements);
    printf("[InplaceRsqrt Tiling] Block: %d elements x %d blocks, Cores: %d\n",
           blockSizeElements, blockCount, coreNum);
}
