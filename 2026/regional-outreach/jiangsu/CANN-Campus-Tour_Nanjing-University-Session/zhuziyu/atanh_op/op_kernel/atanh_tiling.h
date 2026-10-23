/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * atanh_tiling.h
 *
 * Shared tiling structure for the AscendC atanh operator.
 *
 * atanh: one input, one output, 4-D shape [N4, N3, N2, N1], dtype = float16,
 *        format = ND (N1 is the innermost / contiguous dimension).
 *
 * This header is included by BOTH the host-side tiling code (op_host) and the
 * device-side kernel (op_kernel), so it must contain only POD members and be
 * free of any CANN/toolkit dependency.
 */
#ifndef ATANH_TILING_H
#define ATANH_TILING_H

#include <cstdint>

/**
 * TilingData carries the per-launch tiling strategy from host to device.
 *
 * The operator is a pure element-wise unary op, so the only thing that matters
 * for the kernel is how many elements there are in total and how to chunk them.
 * The ND layout + contiguous inner dim means the whole tensor can be treated as
 * one flat 1-D buffer of `totalLength` half elements.
 */
typedef struct {
    uint32_t blockNum;     // number of AI cores to launch for this kernel
    uint32_t totalLength;  // total element count = N4 * N3 * N2 * N1
    uint32_t blockLength;  // base element count assigned to each core (ceil-div)
    uint32_t tileLength;   // element count processed per vector tile
} TilingData;

static_assert(sizeof(TilingData) <= 512,
              "TilingData must fit in the 512-byte tiling buffer");

#endif  // ATANH_TILING_H
