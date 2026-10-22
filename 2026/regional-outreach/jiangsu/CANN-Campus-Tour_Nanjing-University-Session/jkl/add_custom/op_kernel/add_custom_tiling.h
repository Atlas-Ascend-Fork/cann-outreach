/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// add_custom 算子：z = x + y
//   dtype  : float16
//   format : ND
//   shape  : 2 维 [N2, N1]
// 本文件只含纯 C/C++ 语法（不含 __aicore__、__gm__ 等 ASC 关键字），
// 由 kernel 侧和 host 侧共同包含，保证 tiling 计算逻辑只有一份。
// ============================================================================

#pragma once

#include <cstdint>

// [CONFIG] 单次搬运/计算的元素个数，影响 UB 占用
// float16 每元素 2 字节：8192 元素 = 16 KB
// 3 条队列 × 双缓冲 = 6 × 16 KB = 96 KB < UB 192 KB（Atlas A2）
constexpr uint32_t TILE_LENGTH = 8192;

// [CONFIG] 队列深度（双缓冲：MTE2 搬运与 Vector 计算流水并行）
constexpr uint32_t DOUBLE_BUFFER = 2;

// float16 元素字节数；host 侧没有 half 类型，用该常量换算字节数
constexpr uint32_t DTYPE_BYTES = 2;

// Tiling 数据结构 - 向核函数传递的运行时参数
struct AddCustomTilingData {
    uint32_t blockNum;         // 实际使用核数
    uint32_t dim0;             // 输入/输出第 0 维（行数 N2）
    uint32_t dim1;             // 输入/输出第 1 维（列数 N1）
    uint64_t totalLength;      // 总元素数 = dim0 * dim1
    uint64_t numPerCore;       // 除最后一个核外，每个核处理的元素数（TILE_LENGTH 整数倍）
    uint64_t tailNumLastCore;  // 最后一个核处理的元素数
};

// 按核数把 totalLength 切分为 blockNum 段：
//   - 前 blockNum-1 个核各处理 numPerCore 个元素（TILE_LENGTH 整数倍）
//   - 最后一个核处理剩余元素 tailNumLastCore，可能是 TILE_LENGTH 的尾块
// host 直调与 PyTorch 接入共用本函数，避免两份 tiling 逻辑漂移。
inline AddCustomTilingData ComputeAddCustomTiling(uint64_t totalLength, uint32_t dim0, uint32_t dim1,
                                                  uint32_t coreNum)
{
    AddCustomTilingData tiling{};
    tiling.dim0 = dim0;
    tiling.dim1 = dim1;
    tiling.totalLength = totalLength;
    if (totalLength == 0 || coreNum == 0) {
        return tiling;  // 非法输入：blockNum 保持 0，由调用方校验报错
    }

    uint64_t totalTiles = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;
    uint64_t tilesPerCore = (totalTiles + coreNum - 1) / coreNum;
    tiling.blockNum = static_cast<uint32_t>((totalTiles + tilesPerCore - 1) / tilesPerCore);
    tiling.numPerCore = tilesPerCore * TILE_LENGTH;
    tiling.tailNumLastCore = totalLength - tiling.numPerCore * (tiling.blockNum - 1);
    return tiling;
}
