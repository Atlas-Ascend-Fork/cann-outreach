#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const TanhTilingData* tilingData
    )
    {
        const int64_t offset = GetBlockIdx() * tilingData->blockFactor;
        const int64_t remaining = tilingData->totalNum - offset;
        blockLength_ = remaining < tilingData->blockFactor
            ? remaining
            : tilingData->blockFactor;
        ubLength_ = tilingData->ubFactor;
        inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

        pipe.InitBuffer(
            inputQueueX,
            BUFFER_NUM,
            ubLength_ * sizeof(T)
        );
        pipe.InitBuffer(
            outputQueueY,
            BUFFER_NUM,
            ubLength_ * sizeof(T)
        );
        pipe.InitBuffer(expBuffer, ubLength_ * sizeof(float));
        if constexpr (sizeof(T) != sizeof(float)) {
            pipe.InitBuffer(resultBuffer, ubLength_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
            const int64_t remaining = blockLength_ - offset;
            const int64_t count = remaining < ubLength_ ? remaining : ubLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
        DataCopy(xLocal, inputGMX[offset], count);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
        LocalTensor<float> expLocal = expBuffer.Get<float>();
        LocalTensor<float> resultLocal;
        if constexpr (sizeof(T) == sizeof(float)) {
            // float32 直接写输出队列，无需额外结果缓冲区。
            resultLocal = yLocal;
        } else {
            resultLocal = resultBuffer.Get<float>();
        }

        if constexpr (sizeof(T) == sizeof(float)) {
            Muls(expLocal, xLocal, -2.0f, count);
        } else {
            Cast(expLocal, xLocal, RoundMode::CAST_NONE, count);
            Muls(expLocal, expLocal, -2.0f, count);
        }

        // tanh(x) = 2 / (1 + exp(-2x)) - 1。
        // float32 中计算，并限制指数范围；在 |x| >= 10 时已趋于 ±1。
        Maxs(expLocal, expLocal, -20.0f, count);
        Mins(expLocal, expLocal, 20.0f, count);
        Exp(expLocal, expLocal, count);
        Adds(expLocal, expLocal, 1.0f, count);
        Duplicate(resultLocal, 2.0f, count);
        Div(resultLocal, resultLocal, expLocal, count);
        Adds(resultLocal, resultLocal, -1.0f, count);

        if constexpr (sizeof(T) != sizeof(float)) {
            Cast(yLocal, resultLocal, RoundMode::CAST_RINT, count);
        }

        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
        DataCopy(outputGMY[offset], yLocal, count);
        outputQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> expBuffer;
    TBuf<TPosition::VECCALC> resultBuffer;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};
}  // namespace NsTanh
#endif  // TANH_H
