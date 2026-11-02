#ifndef MUL_TILING_H
#define MUL_TILING_H

#include <cstdint>

struct MulTilingData {
    uint32_t length;       // Number of valid elements in the complete tensor.
    uint32_t blockLength;  // Aligned element stride between adjacent cores.
    uint32_t tileLength;   // Element capacity of each input's UB region.
};

#endif
