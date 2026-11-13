/*!
 * \file tanh.h
 * \brief Tanh for the actual judge's three-field tiling structure.
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t MAX_TILE_ELEMS = 2048;

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        blockLength = 0;
        tileLength = 0;
        if (tilingData->totalNum <= 0 || tilingData->blockFactor <= 0 ||
            tilingData->ubFactor <= 0) {
            return;
        }
        const int64_t offset = static_cast<int64_t>(AscendC::GetBlockIdx()) *
                               tilingData->blockFactor;
        if (offset >= tilingData->totalNum) {
            return;
        }
        const int64_t remaining = tilingData->totalNum - offset;
        blockLength = remaining < tilingData->blockFactor
                          ? remaining : tilingData->blockFactor;
        int64_t tile = tilingData->ubFactor < blockLength
                           ? tilingData->ubFactor : blockLength;
        if (tile > MAX_TILE_ELEMS) {
            tile = MAX_TILE_ELEMS;
        }
        tileLength = static_cast<uint32_t>(tile);
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength);

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(tmpA, tileLength * sizeof(float));
        pipe.InitBuffer(tmpB, tileLength * sizeof(float));
        pipe.InitBuffer(tmpC, tileLength * sizeof(float));
        // One comparison bit per element; round the byte allocation to 32 B.
        pipe.InitBuffer(maskBuffer, ((tileLength + 255) / 256) * 32);
        if constexpr (sizeof(DT_X) == sizeof(half)) {
            pipe.InitBuffer(fp32Input, tileLength * sizeof(float));
            pipe.InitBuffer(fp32Output, tileLength * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLength <= 0 || tileLength == 0) {
            return;
        }
        // Use the effective tile length after the UB cap, including in the
        // single-core UT where the supplied ubFactor is the entire tensor.
        for (int64_t offset = 0; offset < blockLength; offset += tileLength) {
            const int64_t remaining = blockLength - offset;
            const uint32_t count = remaining < tileLength
                                       ? static_cast<uint32_t>(remaining) : tileLength;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[offset], count);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ComputeFloat(const LocalTensor<float>& dst,
                                      const LocalTensor<float>& src,
                                      uint32_t count)
    {
        LocalTensor<float> a = tmpA.Get<float>();
        LocalTensor<float> b = tmpB.Get<float>();
        LocalTensor<float> p = tmpC.Get<float>();
        LocalTensor<uint8_t> mask = maskBuffer.Get<uint8_t>();

        // Independent vector operations share a barrier where possible.
        Mins(a, src, 9.0f, count);
        Mul(b, src, src, count);                       // b = x*x
        PipeBarrier<PIPE_V>();

        Maxs(a, a, -9.0f, count);                      // clip before Exp
        Muls(p, b, 2.0f / 15.0f, count);
        CompareScalar(mask, b, 0.015625f, CMPMODE::GE, count);
        PipeBarrier<PIPE_V>();

        Muls(a, a, 2.0f, count);
        Adds(p, p, -1.0f / 3.0f, count);
        PipeBarrier<PIPE_V>();

        Exp(a, a, count);                             // a = exp(2*clip(x))
        Mul(p, p, b, count);
        PipeBarrier<PIPE_V>();

        Adds(p, p, 1.0f, count);
        Adds(b, a, 1.0f, count);                       // b = exp(2x)+1
        PipeBarrier<PIPE_V>();

        Mul(p, p, src, count);                        // p = x-x^3/3+2*x^5/15
        Adds(a, a, -1.0f, count);                      // a = exp(2x)-1
        PipeBarrier<PIPE_V>();

        Div(dst, a, b, count);
        PipeBarrier<PIPE_V>();

        // |x| >= 1/8: exponential formula; otherwise: stable polynomial.
        // GE is false for NaN, selecting the NaN polynomial result as well.
        Select(dst, mask, dst, p, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        if constexpr (sizeof(DT_X) == sizeof(half)) {
            LocalTensor<float> xFloat = fp32Input.Get<float>();
            LocalTensor<float> yFloat = fp32Output.Get<float>();
            Cast(xFloat, xLocal, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            ComputeFloat(yFloat, xFloat, count);
            Cast(yLocal, yFloat, RoundMode::CAST_RINT, count);
        } else {
            ComputeFloat(yLocal, xLocal, count);
        }
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t count)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[offset], yLocal, count);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> tmpA;
    TBuf<QuePosition::VECCALC> tmpB;
    TBuf<QuePosition::VECCALC> tmpC;
    TBuf<QuePosition::VECCALC> maskBuffer;
    TBuf<QuePosition::VECCALC> fp32Input;
    TBuf<QuePosition::VECCALC> fp32Output;
    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;
    int64_t blockLength;
    uint32_t tileLength;
};

} // namespace NsTanh
#endif // TANH_H
