/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * atanh.cpp — AscendC kernel for the atanh (inverse hyperbolic tangent) operator.
 *
 * Spec:
 *   - inputs : x  [N4, N3, N2, N1]  float16  (ND, N1 contiguous)
 *   - output : y  [N4, N3, N2, N1]  float16  (ND, same shape as x)
 *   - math   : y = atanh(x) = 0.5 * ln((1 + x) / (1 - x))
 *
 * Design notes
 * ------------
 * 1. atanh has no single dedicated vector instruction in the basic AscendC math
 *    set, so it is computed from the identity above using well-supported
 *    primitives: Cast, Duplicate, Add, Sub, Div, Ln, Muls.
 *
 * 2. Numerical precision.  The transcendental is evaluated in float32 and the
 *    result is cast back to float16.  Doing it directly in float16 would
 *    overflow the intermediate (1+x)/(1-x) whenever |x| is close to 1, and
 *    would lose accuracy even in the mid range.  Computing in float32 is what a
 *    real atanh kernel does and keeps the result within fp16 rounding error.
 *
 * 3. Layout.  Because the tensor is ND and the inner dim N1 is contiguous, the
 *    whole 4-D tensor is a flat stream of `totalLength` half elements.  The
 *    kernel therefore treats x and y as 1-D and splits that stream across the
 *    AI cores, handling the tail (coreLength) on the last core.
 *
 * 4. Data flow (vector programming paradigm) — three stages per tile:
 *        CopyIn  : DMA x tile  Global -> UB (VECIN queue)
 *        Compute : vector math, UB -> UB (VECCALC queue)
 *        CopyOut : DMA y tile  UB -> Global (VECOUT queue)
 */
#include "kernel_operator.h"
#include "atanh_tiling.h"

using namespace AscendC;

namespace {

// Number of float32 working tensors kept in the VECCALC queue:
//   xF, oneF, pF, mF, rF, lrF, yF
constexpr uint32_t FLOAT_TENSOR_NUM = 7;

// Each AI core processes a contiguous slice of the flat tensor.
class KernelAtanh {
public:
    __aicore__ inline KernelAtanh() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *tilingAddr)
    {
        // Pull the tiling strategy out of the (global) tiling buffer.
        const TilingData *tiling = reinterpret_cast<const TilingData *>(tilingAddr);

        // Per-core slice of the flat [totalLength] element stream.
        totalLength_ = tiling->totalLength;
        tileLength_ = tiling->tileLength;

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockLength = (totalLength_ + blockNum - 1) / blockNum;
        const uint32_t start = blockIdx * blockLength;
        coreLength_ = (start >= totalLength_) ? 0u : (blockLength < (totalLength_ - start) ? blockLength : (totalLength_ - start));

        // Point the global tensors at this core's slice.
        xGm_.SetGlobalBuffer((__gm__ half *)(x + (size_t)start * sizeof(half)), coreLength_);
        yGm_.SetGlobalBuffer((__gm__ half *)(y + (size_t)start * sizeof(half)), coreLength_);

        // Reserve unified-buffer (UB) space.
        //   inQueueX_     : one half tile (input)
        //   computeQueue_ : FLOAT_TENSOR_NUM float tiles (working set)
        //   outQueueY_    : one half tile (output)
        pipe_.InitBuffer(inQueueX_, 1, tileLength_ * sizeof(half));
        pipe_.InitBuffer(computeQueue_, FLOAT_TENSOR_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(outQueueY_, 1, tileLength_ * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        if (coreLength_ == 0) {
            return;  // this core has no elements (totalLength < blockNum)
        }
        for (uint32_t offset = 0; offset < coreLength_; offset += tileLength_) {
            const uint32_t curLen = (tileLength_ < (coreLength_ - offset)) ? tileLength_ : (coreLength_ - offset);
            CopyIn(offset, curLen);
            Compute(offset, curLen);
            CopyOut(offset, curLen);
        }
    }

private:
    // Stage 1: DMA the input tile from Global Memory into UB.
    __aicore__ inline void CopyIn(const uint32_t offset, const uint32_t count)
    {
        LocalTensor<half> xLocal = inQueueX_.AllocTensor<half>();
        DataCopy(xLocal, xGm_[offset], count);
        inQueueX_.EnQue(xLocal);
    }

    // Stage 2: atanh in float32, result stored in a half LocalTensor on VECOUT.
    __aicore__ inline void Compute(const uint32_t offset, const uint32_t count)
    {
        // Input tile (half).
        LocalTensor<half> xLocal = inQueueX_.DeQue<half>();

        // Float32 working tensors.
        LocalTensor<float> xF = computeQueue_.AllocTensor<float>();
        LocalTensor<float> oneF = computeQueue_.AllocTensor<float>();
        LocalTensor<float> pF = computeQueue_.AllocTensor<float>();
        LocalTensor<float> mF = computeQueue_.AllocTensor<float>();
        LocalTensor<float> rF = computeQueue_.AllocTensor<float>();
        LocalTensor<float> lrF = computeQueue_.AllocTensor<float>();
        LocalTensor<float> yF = computeQueue_.AllocTensor<float>();

        // Output tile (half) on the VECOUT queue.
        LocalTensor<half> yLocal = outQueueY_.AllocTensor<half>();

        // x = atanh = 0.5 * ln((1 + x) / (1 - x))
        Cast(xF, xLocal, count);      // float32 = half
        Duplicate(oneF, 1.0f, count); // oneF = 1.0
        Add(pF, oneF, xF, count);     // pF = 1 + x
        Sub(mF, oneF, xF, count);     // mF = 1 - x
        Div(rF, pF, mF, count);       // rF = (1 + x) / (1 - x)
        Ln(lrF, rF, count);           // lrF = ln(rF)
        Muls(yF, lrF, 0.5f, count);   // yF = 0.5 * lrF
        Cast(yLocal, yF, count);      // half = float32

        outQueueY_.EnQue<half>(yLocal);

        // Release tensors for reuse by the next tile.
        inQueueX_.FreeTensor(xLocal);
        computeQueue_.FreeTensor(xF);
        computeQueue_.FreeTensor(oneF);
        computeQueue_.FreeTensor(pF);
        computeQueue_.FreeTensor(mF);
        computeQueue_.FreeTensor(rF);
        computeQueue_.FreeTensor(lrF);
        computeQueue_.FreeTensor(yF);
    }

    // Stage 3: DMA the output tile from UB back to Global Memory.
    __aicore__ inline void CopyOut(const uint32_t offset, const uint32_t count)
    {
        LocalTensor<half> yLocal = outQueueY_.DeQue<half>();
        DataCopy(yGm_[offset], yLocal, count);
        outQueueY_.FreeTensor(yLocal);
    }

    // Per-core state.
    uint32_t totalLength_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t coreLength_ = 0;

    // Memory / pipeline objects.
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECCALC, FLOAT_TENSOR_NUM> computeQueue_;
    TQue<QuePosition::VECOUT, 1> outQueueY_;
    GlobalTensor<half> xGm_;
    GlobalTensor<half> yGm_;
};

}  // namespace

/**
 * Kernel entry point.
 *
 * @param x       input  (float16, ND)
 * @param y       output (float16, ND)
 * @param tiling  pointer to a TilingData in global memory
 */
extern "C" __global__ __aicore__ void atanh(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *tiling)
{
    KernelAtanh op;
    op.Init(x, y, tiling);
    op.Process();
}

// Convenience wrapper used by the CPU debug path / direct kernel launch.
#ifndef ASCENDC_CPU_DEBUG
void atanh_do(uint32_t numBlocks, uint32_t dynUBufSize, void *stream, uint8_t *x, uint8_t *y, uint8_t *tiling)
{
    atanh<<<numBlocks, dynUBufSize, stream>>>(x, y, tiling);
}
#endif
