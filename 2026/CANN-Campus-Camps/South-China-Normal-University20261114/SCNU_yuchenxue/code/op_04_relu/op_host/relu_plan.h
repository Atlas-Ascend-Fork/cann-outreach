#ifndef COURSE_RELU_PLAN_H
#define COURSE_RELU_PLAN_H
#include <cstdint>

namespace CourseRelu {
struct Plan {
    uint32_t totalLength;
    uint32_t blockDim;
    uint32_t tileNum;
    uint32_t tileLength;
    uint64_t usedUbBytes;
};

// 纯 Host 分块计算，供 CANN Tiling 与本地覆盖范围检查共同使用。
inline bool MakePlan(uint32_t length, uint32_t typeBytes, uint32_t availableCores,
                     uint64_t ubBytes, uint32_t buffers, Plan& plan)
{
    if ((length != 16384 && length != 4194304) ||
        (typeBytes != 2 && typeBytes != 4) ||
        availableCores == 0 || (buffers != 1 && buffers != 2)) {
        return false;
    }
    uint32_t blockDim = length <= 16384 ? 8 : 32;
    while (blockDim > availableCores) {
        blockDim /= 2;
    }
    const uint32_t blockLength = length / blockDim;
    // Small tensors need only one copy/compute/store per core.
    if (length <= 16384 && uint64_t(2) * blockLength * typeBytes <= ubBytes) {
        plan = {length, blockDim, 1, blockLength,
                uint64_t(2) * blockLength * typeBytes};
        return true;
    }
    // Keep the established double-buffer UB budget for streaming tensors.
    uint64_t capacity = ubBytes / (2 * buffers * typeBytes * 2);
    if (capacity > blockLength / 2) {
        capacity = blockLength / 2;
    }
    if (capacity == 0) {
        return false;
    }
    uint32_t tileLength = 1;
    while (tileLength <= capacity / 2) {
        tileLength *= 2;
    }
    if ((tileLength * typeBytes) % 32 != 0) {
        return false;
    }
    plan.totalLength = length;
    plan.blockDim = blockDim;
    plan.tileLength = tileLength;
    // tileNum is the actual number of copy/compute/store iterations.
    plan.tileNum = blockLength / tileLength;
    plan.usedUbBytes = uint64_t(2) * buffers * tileLength * typeBytes;
    return plan.tileNum > 0 && plan.usedUbBytes <= ubBytes;
}
} // namespace CourseRelu
#endif
