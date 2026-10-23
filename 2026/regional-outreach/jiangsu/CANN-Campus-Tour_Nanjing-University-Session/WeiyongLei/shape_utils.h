#ifndef INPLACE_RSQRT_SHAPE_UTILS_H
#define INPLACE_RSQRT_SHAPE_UTILS_H
#include <cstddef>
#include <cstdint>
#include <limits>
namespace inplace_rsqrt_host {
constexpr uint32_t TILE_ELEMENTS = 4096;
inline bool ShapeSize(const uint64_t shape[4], uint64_t &elements, size_t &bytes)
{
    elements = 0;
    bytes = 0;
    for (uint32_t axis = 0; axis < 4; ++axis) {
        if (shape[axis] == 0) return true;
    }
    const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<size_t>::max() / 2);
    uint64_t product = 1;
    for (uint32_t axis = 0; axis < 4; ++axis) {
        if (shape[axis] > limit / product) return false;
        product *= shape[axis];
    }
    elements = product;
    bytes = static_cast<size_t>(product) * 2;
    return true;
}
} // namespace inplace_rsqrt_host
#endif
