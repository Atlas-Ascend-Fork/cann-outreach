/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * atanh_tiling.cpp — host-side tiling computation for the AscendC atanh operator.
 *
 * In the CANN operator framework the tiling runs on the host, *before* the
 * kernel is launched.  It inspects the input tensor (shape / dtype / format)
 * and fills a TilingData buffer that is then streamed to the device and read
 * by the kernel (see op_kernel/atanh.cpp).
 *
 * atanh is a pure element-wise unary op, so the only things to decide are:
 *   - how many elements there are in total, and
 *   - how to chunk them across the available AI cores and vector tiles.
 *
 * NOTE: the exact tiling entry signature differs across CANN versions /
 * frameworks (TBE custom op vs. the newer CANN 7/8 tiling API).  The core
 * strategy below, AtanhTiling(), is version-independent and is what the entry
 * point delegates to.
 */
#include "atanh_tiling.h"

#include <algorithm>
#include <cstdint>
#include <cstddef>

namespace {

// Upper bound on AI cores we will use.  Real products expose the core count
// (e.g. 20 on the 910 series); clamp to it at runtime via the SoC info if
// available.  20 is a safe generic cap.
constexpr uint32_t kMaxCoreNum = 20;

// Vector tile size in elements.  Must be a multiple of 8 so that, for float16
// (2 bytes), each tile is 32-byte aligned and a clean multiple of the vector
// unit's 256-byte data block (128 half elements).  512 = 4 x 128.
constexpr uint32_t kTileLength = 512;

// Clamp helper.
inline uint32_t Clamp(uint32_t v, uint32_t lo, uint32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace

/**
 * Fill the TilingData for an atanh kernel launch.
 *
 * @param[in]  shape    pointer to the N dimension sizes, ND order [N4, N3, N2, N1]
 * @param[in]  ndim     number of dimensions (expected 4)
 * @param[out] tiling   output tiling buffer (also streamed to the device)
 *
 * @returns true on success, false if the input is degenerate.
 */
bool AtanhTiling(const uint32_t *shape, uint32_t ndim, TilingData *tiling)
{
    if (shape == nullptr || tiling == nullptr || ndim == 0) {
        return false;
    }

    // 1. Total element count (ND -> contiguous, so a plain product works).
    uint64_t totalLength = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        totalLength *= shape[i];
    }
    if (totalLength == 0) {
        return false;
    }

    // 2. Number of cores: use one core per tile up to the cap.
    const uint32_t idealCores = static_cast<uint32_t>((totalLength + kTileLength - 1) / kTileLength);
    const uint32_t blockNum = Clamp(idealCores, 1, kMaxCoreNum);

    // 3. Base element count per core (ceil-divide so the last core takes the
    //    remainder; the kernel clamps its own slice to totalLength).
    const uint32_t blockLength = static_cast<uint32_t>((totalLength + blockNum - 1) / blockNum);

    // 4. Emit.
    tiling->blockNum = blockNum;
    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->blockLength = blockLength;
    tiling->tileLength = (blockLength < kTileLength) ? blockLength : kTileLength;

    return true;
}

/*
 * ---------------------------------------------------------------------------
 * CANN tiling entry point (illustrative).
 *
 * Replace the body with the real framework signature for your CANN version.
 * A typical custom-op tiling entry looks like:
 *
 *   extern "C" int32_t AtanhTilingFunc(
 *       const OperatorDesc *opDesc,
 *       TilingData *tiling)
 *   {
 *       uint32_t ndim = opDesc->GetInputShapeDim(0);
 *       const uint32_t *shape = opDesc->GetInputShape(0);
 *       return AtanhTiling(shape, ndim, tiling) ? 0 : -1;
 *   }
 *
 * The exact OperatorDesc / TilingContext API depends on the CANN release.
 * ---------------------------------------------------------------------------
 */
