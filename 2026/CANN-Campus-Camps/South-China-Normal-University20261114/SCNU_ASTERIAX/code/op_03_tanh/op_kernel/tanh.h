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
    TBuf<QuePosition::VECCALC> positiveBuffer;
    TBuf<QuePosition::VECCALC> negativeBuffer;

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
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(valueBuffer, ubLength_ * sizeof(float));
    pipe.InitBuffer(positiveBuffer, ubLength_ * sizeof(float));
    pipe.InitBuffer(negativeBuffer, ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const uint32_t count = static_cast<uint32_t>(currentNum);
    if (count * sizeof(T) % 32 == 0) {
        DataCopy(xLocal, inputGMX[progress], count);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> value = valueBuffer.Get<float>();
    LocalTensor<float> positive = positiveBuffer.Get<float>();
    LocalTensor<float> negative = negativeBuffer.Get<float>();
    const uint32_t count = static_cast<uint32_t>(currentNum);
    if constexpr (sizeof(T) == sizeof(half)) {
        Cast(value, xLocal, RoundMode::CAST_NONE, count);
    } else {
        Adds(value, xLocal, 0.0f, count);
    }

    // Beyond this range tanh has already rounded to +/-1 in both dtypes.
    // Clamping keeps both exponentials finite, including for infinite input.
    Mins(value, value, 44.0f, count);
    Maxs(value, value, -44.0f, count);
    Exp(positive, value, count);
    Muls(value, value, -1.0f, count);
    Exp(negative, value, count);
    Sub(value, positive, negative, count);
    Add(negative, positive, negative, count);
    Div(positive, value, negative, count);

    if constexpr (sizeof(T) == sizeof(half)) {
        Cast(yLocal, positive, RoundMode::CAST_ROUND, count);
    } else {
        Adds(yLocal, positive, 0.0f, count);
    }
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const uint32_t count = static_cast<uint32_t>(currentNum);
    if (count * sizeof(T) % 32 == 0) {
        DataCopy(outputGMY[progress], yLocal, count);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[progress], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
        const int64_t remaining = blockLength_ - offset;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
