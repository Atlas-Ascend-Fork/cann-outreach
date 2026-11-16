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
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (offset >= tilingData->totalNum) {
        return;
    }
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
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
    if (currentNum * sizeof(T) % 32 == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        const uint32_t alignElements = 32 / sizeof(T);
        const uint8_t pad = static_cast<uint8_t>(alignElements - currentNum % alignElements);
        DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padding{true, 0, pad, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], params, padding);
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
    const uint32_t count = static_cast<uint32_t>(currentNum);

    // tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1).
    // Clamp before Exp: outside [-10, 10], the saturation error is < 4.2e-9.
    // Float16 inputs use fp32 intermediate arithmetic to avoid fp16 overflow.
    if constexpr (sizeof(T) == sizeof(float)) {
        Maxs(value, xLocal, -10.0f, count);
    } else {
        Cast(value, xLocal, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Maxs(value, value, -10.0f, count);
    }
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
    if constexpr (sizeof(T) == sizeof(float)) {
        Div(yLocal, value, denominator, count);
    } else {
        Div(value, value, denominator, count);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, value, RoundMode::CAST_RINT, count);
    }
    // Scratch buffers are shared between tiles; finish reads before reusing them.
    PipeBarrier<PIPE_V>();
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    if (currentNum * sizeof(T) % 32 == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, params);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }
    const int64_t loops = blockLength_ / ubLength_ + (blockLength_ % ubLength_ != 0);
    for (int64_t progress = 0; progress < loops; ++progress) {
        const int64_t remaining = blockLength_ - progress * ubLength_;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
