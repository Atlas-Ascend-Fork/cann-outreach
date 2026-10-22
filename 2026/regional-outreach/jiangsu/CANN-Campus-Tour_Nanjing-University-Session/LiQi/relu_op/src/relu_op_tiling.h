/**
 * @file relu_op_tiling.h
 * @brief Tiling data structure and interface for ReLU operator
 */

#ifndef RELU_OP_TILING_H
#define RELU_OP_TILING_H

#include <cstdint>

/**
 * @struct ReluTilingData
 * @brief Tiling parameters shared between host and kernel
 */
struct ReluTilingData {
    int64_t totalElements;      // Total elements: N4 * N3 * N2 * N1
    int32_t blockCount;         // Number of processing blocks
    int32_t blockSizeElements;  // Elements per block
};

/**
 * @struct ReluTiling
 * @brief Host-side tiling configuration for ReLU
 */
struct ReluTiling {
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
     * @param ubSizeBytes Available UB size in bytes
     * @param coreNum Number of AI cores available
     */
    void CalcTiling(int32_t ubSizeBytes = 192 * 1024, int32_t coreNum = 20);
};

#endif  // RELU_OP_TILING_H
