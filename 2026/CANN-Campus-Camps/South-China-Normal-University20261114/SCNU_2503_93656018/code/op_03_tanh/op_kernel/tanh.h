/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include <type_traits>
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

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    TBuf<QuePosition::VECCALC> valueBuffer;
    TBuf<QuePosition::VECCALC> denominatorBuffer;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t offset = GetBlockIdx() * tilingData->blockFactor;
    blockLength_ = tilingData->totalNum - offset;
    if (blockLength_ > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    }
    if (blockLength_ <= 0) {
        return;
    }
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
    auto xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    const uint8_t padding = static_cast<uint8_t>((32 - currentNum * sizeof(T) % 32) % 32 / sizeof(T));
    DataCopyPadExtParams<T> padParams{true, 0, padding, 0};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    auto xLocal = inputQueueX.DeQue<T>();
    auto yLocal = outputQueueY.AllocTensor<T>();
    auto value = valueBuffer.Get<float>();
    auto denominator = denominatorBuffer.Get<float>();
    if constexpr (std::is_same<T, half>::value) {
        Cast(value, xLocal, RoundMode::CAST_NONE, currentNum);
    } else {
        Adds(value, xLocal, 0.0f, currentNum);
    }
    PipeBarrier<PIPE_V>();
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)。限幅避免指数溢出。
    Mins(value, value, 10.0f, currentNum);
    PipeBarrier<PIPE_V>();
    Maxs(value, value, -10.0f, currentNum);
    PipeBarrier<PIPE_V>();
    Muls(value, value, 2.0f, currentNum);
    PipeBarrier<PIPE_V>();
    Exp(value, value, currentNum);
    PipeBarrier<PIPE_V>();
    Adds(denominator, value, 1.0f, currentNum);
    PipeBarrier<PIPE_V>();
    Adds(value, value, -1.0f, currentNum);
    PipeBarrier<PIPE_V>();
    Div(value, value, denominator, currentNum);
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same<T, half>::value) {
        Cast(yLocal, value, RoundMode::CAST_RINT, currentNum);
    } else {
        Adds(yLocal, value, 0.0f, currentNum);
    }
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    auto yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    for (int64_t offset = 0, progress = 0; offset < blockLength_; offset += ubLength_, ++progress) {
        const int64_t currentNum = blockLength_ - offset < ubLength_ ? blockLength_ - offset : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H

