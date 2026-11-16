#pragma once
#include <cstdint>

// Change only these constants to tune the number of cores.
// Supported choices: 4, 8, 16, 32. Rebuild both host and kernel.
constexpr uint32_t MUL_FP16_CORES = 4;
constexpr uint32_t MUL_FP32_CORES = 4;
constexpr uint32_t MUL_TOTAL_LENGTH = 8 * 2048;
static_assert(MUL_FP16_CORES == 4 || MUL_FP16_CORES == 8 ||
              MUL_FP16_CORES == 16 || MUL_FP16_CORES == 32, "Invalid FP16 core count");
static_assert(MUL_FP32_CORES == 4 || MUL_FP32_CORES == 8 ||
              MUL_FP32_CORES == 16 || MUL_FP32_CORES == 32, "Invalid FP32 core count");

// Retained for the framework's tiling registration; not read by the kernel.
struct MulTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
};

