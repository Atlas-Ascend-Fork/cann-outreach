/**
 * add_custom 算子主机侧封装（CPU 调度，ACLNN/单算子模型调用前的基础封装）
 * 职责：计算 tiling、设置核函数启动参数并调用 add_custom 核函数
 */
#include "add_custom_tiling.h"
#include "kernel_operator.h" // host 侧仅需宏定义

#include <cstdint>

using namespace AscendC;

constexpr uint32_t TILE_LENGTH = 2048; // 每次搬运/计算的元素个数（half 下为 4KB，满足 32B 对齐）

// shape 以 [N2, N1] 传入，ND 格式下内部按一维展平处理
extern "C" void add_custom_do(uint32_t blockDim, void *l2ctrl, void *stream,
                              const uint8_t *x, const uint8_t *y, uint8_t *z,
                              const AddCustomTilingData *tiling)
{
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, nullptr, tiling);
}

// host 侧 tiling 计算：totalLength = N2 * N1
uint32_t ComputeTiling(uint32_t totalLength, AddCustomTilingData &tiling)
{
    tiling.totalLength = totalLength;
    tiling.tileLength = TILE_LENGTH;
    // 按 tileLength 切分，向上取整得到所需 AI Core 数量
    return (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;
}
