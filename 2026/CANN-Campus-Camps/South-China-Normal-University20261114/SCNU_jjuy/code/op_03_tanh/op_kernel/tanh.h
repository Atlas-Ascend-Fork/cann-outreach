/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> valueBuffer;
    TBuf<QuePosition::VECCALC> denominatorBuffer;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (blockOffset >= tilingData->totalNum || ubLength_ <= 0) {
        return;
    }
    const int64_t remaining = tilingData->totalNum - blockOffset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(valueBuffer, ubLength_ * sizeof(float));
    pipe.InitBuffer(denominatorBuffer, ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    const uint32_t copyBytes = static_cast<uint32_t>(currentNum * sizeof(T));
    if (copyBytes % 32 == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> value = valueBuffer.Get<float>();
    LocalTensor<float> denominator = denominatorBuffer.Get<float>();
    const int32_t count = static_cast<int32_t>(currentNum);

    // half 输入先升精度，避免半精度 Exp 和相减造成较大的误差。
    if constexpr (sizeof(T) == sizeof(half)) {
        Cast(value, xLocal, RoundMode::CAST_NONE, count);
    } else {
        Adds(value, xLocal, 0.0f, count);
    }
    PipeBarrier<PIPE_V>();

    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)。
    // |x| >= 10 时结果已经接近 +/-1，裁剪可防止指数溢出。
    Maxs(value, value, -10.0f, count);
    PipeBarrier<PIPE_V>();
    Mins(value, value, 10.0f, count);
    PipeBarrier<PIPE_V>();
    Muls(value, value, 2.0f, count);
    PipeBarrier<PIPE_V>();
    Exp(value, value, count);
    PipeBarrier<PIPE_V>();
    Adds(denominator, value, 1.0f, count);
    PipeBarrier<PIPE_V>();
    Adds(value, value, -1.0f, count);
    PipeBarrier<PIPE_V>();
    Div(value, value, denominator, count);
    PipeBarrier<PIPE_V>();

    if constexpr (sizeof(T) == sizeof(half)) {
        Cast(yLocal, value, RoundMode::CAST_NONE, count);
    } else {
        Adds(yLocal, value, 0.0f, count);
    }
    PipeBarrier<PIPE_V>();
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    const uint32_t copyBytes = static_cast<uint32_t>(currentNum * sizeof(T));
    if (copyBytes % 32 == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t progress = 0; progress < loopCount; ++progress) {
        const int64_t remaining = blockLength_ - progress * ubLength_;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
