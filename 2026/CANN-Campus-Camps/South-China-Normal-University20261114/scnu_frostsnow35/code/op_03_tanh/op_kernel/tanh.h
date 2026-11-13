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
constexpr int64_t MAX_UB_FACTOR = 2048;  // 单次 UB 计算元素上限，限制临时 buffer 占用
constexpr float EXP_ARG_BOUND = 40.0f;   // 2x 截断到 ±40：exp(40) 不溢出 fp32，|x|>20 时 tanh 已饱和

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

    // fp32 中间量：输入/结果、2x 与分子、exp(2x)、分母
    TBuf<TPosition::VECCALC> xFloatBuf;
    TBuf<TPosition::VECCALC> numeratorBuf;
    TBuf<TPosition::VECCALC> expBuf;
    TBuf<TPosition::VECCALC> denominatorBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remainingNum = tilingData->totalNum - blockOffset;
    blockLength_ = remainingNum < tilingData->blockFactor ? remainingNum : tilingData->blockFactor;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ > MAX_UB_FACTOR) {
        ubLength_ = MAX_UB_FACTOR;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(xFloatBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(numeratorBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(expBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(denominatorBuf, ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    const uint32_t count = static_cast<uint32_t>(currentNum);

    LocalTensor<float> xFloat = xFloatBuf.Get<float>();
    LocalTensor<float> numerator = numeratorBuf.Get<float>();
    LocalTensor<float> expValue = expBuf.Get<float>();
    LocalTensor<float> denominator = denominatorBuf.Get<float>();

    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)，全程 fp32 计算
    if constexpr (sizeof(T) == sizeof(float)) {
        Muls(numerator, xLocal, 2.0f, count);
    } else {
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, count);
        Muls(numerator, xFloat, 2.0f, count);
    }
    Maxs(xFloat, numerator, -EXP_ARG_BOUND, count);
    Mins(numerator, xFloat, EXP_ARG_BOUND, count);
    Exp(expValue, numerator, count);
    Adds(denominator, expValue, 1.0f, count);
    Adds(numerator, expValue, -1.0f, count);
    if constexpr (sizeof(T) == sizeof(float)) {
        Div(yLocal, numerator, denominator, count);
    } else {
        Div(xFloat, numerator, denominator, count);
        Cast(yLocal, xFloat, RoundMode::CAST_RINT, count);
    }

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ == 0 || ubLength_ == 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t progress = 0; progress < loopCount; ++progress) {
        const int64_t offset = progress * ubLength_;
        const int64_t remainingNum = blockLength_ - offset;
        const int64_t currentNum = remainingNum < ubLength_ ? remainingNum : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
