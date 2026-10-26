/*!
 * \file exp_tiling_data.h
 * \brief tiling data struct
 *
 * Exp 算子的 tiling 数据。
 * 字段名与官方单测 tests/ut/op_kernel/test_exp.cpp 中使用的
 * tilingData->totalNum / blockFactor / ubFactor 保持一致。
 */

#ifndef _EXP_TILING_DATA_H_
#define _EXP_TILING_DATA_H_

#include <cstdint> // int64_t：不依赖调用方的包含顺序

struct ExpTilingData {
    int64_t totalNum = 0;    // 算子输入的总元素个数
    int64_t blockFactor = 1; // 每个核负责处理的元素个数（按 32 字节对齐）
    int64_t ubFactor = 0;    // 单个 tile 的元素个数（UB 循环粒度，按 32 字节对齐）
};

#endif
