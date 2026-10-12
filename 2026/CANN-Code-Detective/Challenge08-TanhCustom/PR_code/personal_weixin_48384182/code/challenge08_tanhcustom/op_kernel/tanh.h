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

constexpr int32_t BUFFER_NUM = 1;
constexpr float TANH_CLAMP_MAX = 40.0f;

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
    TBuf<TPosition::VECCALC> calcBuf0;
    TBuf<TPosition::VECCALC> calcBuf1;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t startOffset = blockIdx * tilingData->blockFactor;
    blockLength_ = tilingData->totalNum - startOffset;
    if (blockLength_ > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    }
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor / BUFFER_NUM;
    if (ubLength_ < 1) {
        ubLength_ = 1;
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(calcBuf0, static_cast<uint32_t>(ubLength_ * sizeof(float)));
    if constexpr (sizeof(T) == sizeof(half)) {
        pipe.InitBuffer(calcBuf1, static_cast<uint32_t>(ubLength_ * sizeof(float)));
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    const int32_t calCount = static_cast<int32_t>(currentNum);
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    if constexpr (sizeof(T) == sizeof(half)) {
        LocalTensor<float> expLocal = calcBuf0.Get<float>();
        LocalTensor<float> tLocal = calcBuf1.Get<float>();
        Cast(tLocal, xLocal, RoundMode::CAST_NONE, calCount);
        Mins(tLocal, tLocal, TANH_CLAMP_MAX, calCount);
        Muls(tLocal, tLocal, 2.0f, calCount);
        Exp(expLocal, tLocal, calCount);
        Adds(tLocal, expLocal, -1.0f, calCount);
        Adds(expLocal, expLocal, 1.0f, calCount);
        Div(tLocal, tLocal, expLocal, calCount);
        Cast(yLocal, tLocal, RoundMode::CAST_RINT, calCount);
    } else {
        LocalTensor<float> expLocal = calcBuf0.Get<float>();
        Mins(xLocal, xLocal, static_cast<T>(TANH_CLAMP_MAX), calCount);
        Muls(xLocal, xLocal, static_cast<T>(2.0f), calCount);
        Exp(expLocal, xLocal, calCount);
        Adds(xLocal, expLocal, static_cast<T>(-1.0f), calCount);
        Adds(expLocal, expLocal, static_cast<T>(1.0f), calCount);
        Div(yLocal, xLocal, expLocal, calCount);
    }
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    // TODO: 实现 Process 逻辑
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = blockLength_ - i * ubLength_;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} 
#endif 