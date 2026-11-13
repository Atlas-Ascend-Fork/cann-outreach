#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "leakyrelu_tiling_data.h"
#include "leakyrelu_tiling_key.h"

namespace NsLeakyrelu {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Leakyrelu {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
    {
        const int64_t start = GetBlockIdx() * tilingData->blockFactor;
        const int64_t remaining = tilingData->totalNum - start;
        blockLength = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        ubLength = tilingData->ubFactor;
        negativeSlope = tilingData->negativeSlope;
        inputGM.SetGlobalBuffer((__gm__ T*)x + start, blockLength);
        outputGM.SetGlobalBuffer((__gm__ T*)y + start, blockLength);
        pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength * sizeof(T));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength * sizeof(T));
        pipe.InitBuffer(zeroOrScaledBuf, ubLength * sizeof(float));
        pipe.InitBuffer(negativeBuf, ubLength * sizeof(float));
        pipe.InitBuffer(positiveBuf, ubLength * sizeof(float));
        if constexpr (sizeof(T) < sizeof(float)) {
            pipe.InitBuffer(castBuf, ubLength * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        for (int64_t offset = 0; offset < blockLength; offset += ubLength) {
            const int64_t remaining = blockLength - offset;
            const int32_t count = remaining < ubLength ? remaining : ubLength;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int32_t count)
    {
        LocalTensor<T> x = inputQueue.AllocTensor<T>();
        DataCopy(x, inputGM[offset], count);
        inputQueue.EnQue(x);
    }

    __aicore__ inline void Compute(int32_t count)
    {
        LocalTensor<T> x = inputQueue.DeQue<T>();
        LocalTensor<T> y = outputQueue.AllocTensor<T>();
        if constexpr (sizeof(T) < sizeof(float)) {
            // Keep the slope in FP32 and round only the final FP16 result.
            LocalTensor<float> wide = castBuf.Get<float>();
            Cast(wide, x, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            ComputeFloat(wide, wide, count);
            PipeBarrier<PIPE_V>();
            Cast(y, wide, RoundMode::CAST_RINT, count);
        } else {
            ComputeFloat(y, x, count);
        }
        outputQueue.EnQue(y);
        inputQueue.FreeTensor(x);
    }

    __aicore__ inline void ComputeFloat(LocalTensor<float> y, LocalTensor<float> x, int32_t count)
    {
        LocalTensor<float> zeroOrScaled = zeroOrScaledBuf.Get<float>();
        LocalTensor<float> negative = negativeBuf.Get<float>();
        LocalTensor<float> positive = positiveBuf.Get<float>();
        Duplicate(zeroOrScaled, 0.0f, count);
        PipeBarrier<PIPE_V>();
        Max(positive, x, zeroOrScaled, count);
        Min(negative, x, zeroOrScaled, count);
        PipeBarrier<PIPE_V>();
        Muls(zeroOrScaled, negative, negativeSlope, count);
        PipeBarrier<PIPE_V>();
        Add(y, positive, zeroOrScaled, count);
    }

    __aicore__ inline void CopyOut(int64_t offset, int32_t count)
    {
        LocalTensor<T> y = outputQueue.DeQue<T>();
        DataCopy(outputGM[offset], y, count);
        outputQueue.FreeTensor(y);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<TPosition::VECCALC> zeroOrScaledBuf;
    TBuf<TPosition::VECCALC> negativeBuf;
    TBuf<TPosition::VECCALC> positiveBuf;
    TBuf<TPosition::VECCALC> castBuf;
    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;
    int64_t blockLength;
    int64_t ubLength;
    float negativeSlope;
};

} // namespace NsLeakyrelu
#endif
