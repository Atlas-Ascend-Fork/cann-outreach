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

#include <type_traits>

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
    TBuf<QuePosition::VECCALC> expMinBuf, expMaxBuf, numeratorBuf, denominatorBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t blockOffset = tilingData->blockFactor * AscendC::GetBlockIdx();
    const int64_t remaining = tilingData->totalNum - blockOffset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    const uint32_t tempBytes = ubLength_ * sizeof(float);
    pipe.InitBuffer(expMinBuf, tempBytes);
    pipe.InitBuffer(expMaxBuf, tempBytes);
    pipe.InitBuffer(numeratorBuf, tempBytes);
    pipe.InitBuffer(denominatorBuf, tempBytes);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> expMin = expMinBuf.Get<float>();
    LocalTensor<float> expMax = expMaxBuf.Get<float>();
    LocalTensor<float> numerator = numeratorBuf.Get<float>();
    LocalTensor<float> denominator = denominatorBuf.Get<float>();

    LocalTensor<float> source = numerator;
    const int32_t count = static_cast<int32_t>(currentNum);
    if constexpr (std::is_same_v<T, half>) {
        AscendC::Cast(source, xLocal, RoundMode::CAST_NONE, count);
    } else {
        source = xLocal;
    }

    // 同时缩放 exp(x) 和 exp(-x)，让两个指数的输入都不大于 0。
    // a = exp(2 * min(x, 0)), b = exp(-2 * max(x, 0)), tanh(x) = (a - b) / (a + b)。
    AscendC::Mins(expMin, source, 0.0f, count);
    AscendC::Muls(denominator, expMin, 2.0f, count);
    AscendC::Exp(expMin, denominator, count);
    AscendC::Maxs(expMax, source, 0.0f, count);
    AscendC::Muls(denominator, expMax, -2.0f, count);
    AscendC::Exp(expMax, denominator, count);
    AscendC::Sub(numerator, expMin, expMax, count);
    AscendC::Add(denominator, expMin, expMax, count);

    if constexpr (std::is_same_v<T, half>) {
        AscendC::Div(expMin, numerator, denominator, count);
        AscendC::Cast(yLocal, expMin, RoundMode::CAST_RINT, count);
    } else {
        AscendC::Div(yLocal, numerator, denominator, count);
    }
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
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
