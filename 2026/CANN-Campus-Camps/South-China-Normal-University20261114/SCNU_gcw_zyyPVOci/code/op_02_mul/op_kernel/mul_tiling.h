#ifndef MUL_TILING_H
#define MUL_TILING_H

#include <cstdint>

namespace optiling {
struct MulTilingData {
    uint32_t length;        // 总元素个数
    uint32_t tile_length;    // 单次搬进 UB 的元素个数
};
}  // namespace optiling

#endif  // MUL_TILING_H