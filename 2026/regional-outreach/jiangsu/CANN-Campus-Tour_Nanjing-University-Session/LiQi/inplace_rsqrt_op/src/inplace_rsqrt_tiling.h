/**
 * @file inplace_rsqrt_tiling.h
 * @brief Tiling data structure and interface for InplaceRsqrt operator
 */

#ifndef INPLACE_RSQRT_TILING_H
#define INPLACE_RSQRT_TILING_H

#include <cstdint>

/**
 * @struct RsqrtTilingData
 * @brief Tiling parameters shared between host and kernel
 */
struct RsqrtTilingData {
    int64_t totalElements;      // Total elements: N4 * N3 * N2 * N1
    int32_t blockCount;         // Number of processing blocks
    int32_t blockSizeElements;  // Elements per block (32-byte aligned)
};

/**
 * @struct RsqrtTiling
 * @brief Host-side tiling configuration for InplaceRsqrt
 */
struct RsqrtTiling {
    // Input shape (4D: [N4, N3, N2, N1])
    int64_t n4;
    int64_t n3;
    int64_t n2;
    int64_t n1;

    // Data type (1 = FP16)
    int32_t dataType;

    // Format (0 = ND)
    int32_t format;

    // Computed tiling parameters
    int64_t totalElements;
    int32_t blockCount;
    int32_t blockSizeElements;

    /**
     * @brief Calculate tiling parameters
     *
     * In-place optimization: only 1 UB buffer needed (vs 2 for non-in-place),
     * so block size = UB_SIZE / 2 (bytes per element) — double the throughput
     * of a separate in/out op.
     *
     * @param ubSizeBytes Available UB size per core in bytes
     * @param coreNum Number of AI cores
     */
    void CalcTiling(int32_t ubSizeBytes = 192 * 1024, int32_t coreNum = 20);
};

#endif  // INPLACE_RSQRT_TILING_H
