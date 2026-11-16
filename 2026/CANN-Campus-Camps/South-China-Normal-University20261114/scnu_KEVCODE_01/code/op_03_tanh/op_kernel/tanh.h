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

// 【优化点】：开启 Double Buffer，加速流水线
constexpr int32_t BUFFER_NUM = 2; 

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
        const int64_t offset = static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->blockFactor;
        if (offset >= tilingData->totalNum) {
            return;
        }
        const int64_t remaining = tilingData->totalNum - offset;
        blockLength = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        
        tileLength = tilingData->ubFactor;
        
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength);

        // 使用双缓冲初始化队列
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        
        pipe.InitBuffer(tmpA, tileLength * sizeof(float));
        pipe.InitBuffer(tmpB, tileLength * sizeof(float));
        pipe.InitBuffer(tmpC, tileLength * sizeof(float));
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
        // 按照 tileLength 切块循环，配合 Double Buffer
        int32_t loopCount = (blockLength + tileLength - 1) / tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            const int64_t offset = i * tileLength;
            const int64_t remaining = blockLength - offset;
            const uint32_t count = remaining < tileLength ? static_cast<uint32_t>(remaining) : tileLength;
            
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