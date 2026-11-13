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

    TBuf<TPosition::VECCALC> expXBuf_;
    TBuf<TPosition::VECCALC> expNegXBuf_;
    TBuf<TPosition::VECCALC> numerBuf_;
    TBuf<TPosition::VECCALC> castBuf_;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t offset = blockIdx * tilingData->blockFactor;
    blockLength_ = tilingData->totalNum - offset;
    if (blockLength_ > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    }
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);

    uint32_t inQueueBytes = static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, inQueueBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, inQueueBytes);
    pipe.InitBuffer(expXBuf_, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(float))));
    pipe.InitBuffer(expNegXBuf_, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(float))));
    pipe.InitBuffer(numerBuf_, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(float))));
    if constexpr (sizeof(T) != sizeof(float)) {
        pipe.InitBuffer(castBuf_, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(float))));
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    LocalTensor<float> expX = expXBuf_.Get<float>();
    LocalTensor<float> expNegX = expNegXBuf_.Get<float>();
    LocalTensor<float> numer = numerBuf_.Get<float>();

    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    // 仅实例化 float / half：float 直接计算，half 升精度到 float 计算
    if constexpr (sizeof(T) == sizeof(float)) {
        Exp(expX, xLocal, currentNum);
        Muls(expNegX, xLocal, static_cast<float>(-1.0), currentNum);
        Exp(expNegX, expNegX, currentNum);
        Sub(numer, expX, expNegX, currentNum);
        Add(expX, expX, expNegX, currentNum);
        Div(yLocal, numer, expX, currentNum);
    } else {
        // 半精度输入先升精度到 float 计算，避免 exp(x) - exp(-x) 的精度损失
        LocalTensor<float> xFloat = castBuf_.Get<float>();
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, currentNum);
        Exp(expX, xFloat, currentNum);
        Muls(expNegX, xFloat, static_cast<float>(-1.0), currentNum);
        Exp(expNegX, expNegX, currentNum);
        Sub(numer, expX, expNegX, currentNum);
        Add(expX, expX, expNegX, currentNum);
        Div(numer, numer, expX, currentNum);
        Cast(yLocal, numer, RoundMode::CAST_RINT, currentNum);
    }

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
