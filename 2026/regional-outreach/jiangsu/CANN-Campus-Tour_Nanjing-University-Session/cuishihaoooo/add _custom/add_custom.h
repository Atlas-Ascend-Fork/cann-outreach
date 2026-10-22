/*
 * \file add_custom.h
 * \brief AddCustom —— 逐元素加法算子（Ascend C / AI Core Vector）声明
 *
 * 算子规格
 *   OpType  : AddCustom
 *   输入    : x, 2D [N2, N1], ND, float16 (half)
 *             y, 2D [N2, N1], ND, float16 (half)
 *   输出    : z, 2D [N2, N1], ND, float16 (half)
 *   功能    : z[i] = x[i] + y[i]
 *
 * 本文件只包含声明（算子类 + kernel 入口原型），实现见 add_custom.cpp。
 */

#ifndef ADD_CUSTOM_H
#define ADD_CUSTOM_H

#include "kernel_operator.h"

#include <cstdint>

using namespace AscendC;

// 每个 tile 处理的 half 元素个数（批量向量化粒度）
constexpr uint32_t ADD_CUSTOM_TILE_LEN = 256;

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}

    // 初始化：完成按核数据切分与 UB 缓冲初始化
    __aicore__ void Init(__gm__ uint8_t *xGm, __gm__ uint8_t *yGm, __gm__ uint8_t *zGm,
                         uint32_t totalSize);

    // 主处理：tile 循环 + 三段流水（CopyIn / Compute / CopyOut）
    __aicore__ void Process();

private:
    // CopyIn：x、y 两个输入 tile（GM -> UB VECIN）
    __aicore__ void CopyIn(uint32_t offset, uint32_t len);

    // Compute：批量向量化逐元素加法（等价 vadd 向量指令）
    __aicore__ void Compute(uint32_t len);

    // CopyOut：z（UB VECOUT -> GM）
    __aicore__ void CopyOut(uint32_t offset, uint32_t len);

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQueue_;  // 深度 2：同一时刻同时持有 x、y 两个 tile
    TQue<TPosition::VECOUT, 1> outQueue_;
    GlobalTensor<half> xGlobal_;
    GlobalTensor<half> yGlobal_;
    GlobalTensor<half> zGlobal_;

    uint32_t blockNum_ = 1;
    uint32_t blockIdx_ = 0;
    uint32_t blockLen_ = 0;
    uint32_t inOffset_ = 0;
};

// Kernel 入口
//   x : 输入张量 GM 地址，2D [N2, N1]，ND，half
//   y : 输入张量 GM 地址，2D [N2, N1]，ND，half
//   z : 输出张量 GM 地址，2D [N2, N1]，ND，half
//   totalSize : 总元素个数 = N2 * N1
__global__ __vector__ void AddCustom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z,
                                     uint32_t totalSize);

#endif  // ADD_CUSTOM_H
