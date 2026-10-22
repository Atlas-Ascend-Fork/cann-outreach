// Copyright (C) 2026.
// Gcd123 TilingData：Host(Tiling) 与 Kernel 共享的结构体。
//
// 约定：
//  - 所有形状为 4 维，ND 布局 [N4, N3, N2, N1]，其中 N1 是变化最快的维（连续维）。
//  - 元素总数需能放入 uint32_t（即总元素数 < 4.29e9），否则需扩展为 64 位。
//
#ifndef GCD123_TILING_H
#define GCD123_TILING_H

#include <cstdint>

typedef struct Gcd123TilingData {
    // self 形状 [N4, N3, N2, N1]
    uint32_t selfN4;
    uint32_t selfN3;
    uint32_t selfN2;
    uint32_t selfN1;
    // other 形状 [N4, N3, N2, N1]
    uint32_t otherN4;
    uint32_t otherN3;
    uint32_t otherN2;
    uint32_t otherN1;
    // 输出(broadcast 后)形状 [N4, N3, N2, N1]
    uint32_t outN4;
    uint32_t outN3;
    uint32_t outN2;
    uint32_t outN1;

    // 派生量（由 Host 计算）
    uint32_t totalSelf;    // selfN4 * selfN3 * selfN2 * selfN1
    uint32_t totalOther;   // otherN4 * otherN3 * otherN2 * otherN1
    uint32_t totalOut;     // outN4 * outN3 * outN2 * outN1
    uint32_t totalLines;   // outN4 * outN3 * outN2（沿 N1 的“行”数）
    uint32_t chunkLen;     // 每行内分块处理的元素数（0 表示 Kernel 使用默认 1024）
} Gcd123TilingData;

#endif // GCD123_TILING_H
