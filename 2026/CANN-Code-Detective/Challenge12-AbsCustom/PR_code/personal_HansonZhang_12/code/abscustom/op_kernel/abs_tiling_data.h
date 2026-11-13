/*!
 * \file abs_tiling_data.h
 * \brief tiling data struct
 *
 * Abs 算子的 tiling 数据。
 * 字段名与官方单测 tests/ut/op_kernel/test_abs.cpp 中使用的
 * tilingData->totalNum / blockFactor / ubFactor 保持一致。
 *
 * 默认值沿用官方模板「初始实现」的取值（totalNum=0 / blockFactor=1 / ubFactor=0），
 * 与第八期（Tanh）、第十期（Exp）已合入实现保持同一写法；
 * 运行期由 op_host/abs_tiling.cpp 覆盖这三个字段。
 *
 * 三个字段都是 int64_t：
 *   - kernel 侧用 int64_t 承接 offset / blockLength，避免元素个数较大时溢出；
 *   - host 侧官方单测把 tiling buffer 按 int64_t 数组序列化，
 *     3 × int64_t = 24 字节恰好与 sizeof(AbsTilingData) 一致。
 */

#ifndef _ABS_TILING_DATA_H_
#define _ABS_TILING_DATA_H_

#include <cstdint> // int64_t：不依赖调用方的包含顺序

struct AbsTilingData {
    int64_t totalNum = 0;    // 算子输入的总元素个数
    int64_t blockFactor = 1; // 每个核负责处理的元素个数（按 32 字节 DMA 粒度取整）
    int64_t ubFactor = 0;    // 单个 tile 的元素个数（UB 循环粒度，按 32 字节 DMA 粒度取整）
};

#endif
