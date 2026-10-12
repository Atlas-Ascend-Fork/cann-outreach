/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr float EXP_LIMIT = 80.0f;

template <typename T>
class Sigmoid {
public:
    __aicore__ inline Sigmoid(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> calcQueue;
    TBuf<QuePosition::VECCALC> resultQueue;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Sigmoid<T>::Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t startOffset = blockIdx * tilingData->blockFactor;
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
    pipe.InitBuffer(calcQueue, static_cast<uint32_t>(ubLength_ * sizeof(float)));
    pipe.InitBuffer(resultQueue, static_cast<uint32_t>(ubLength_ * sizeof(float)));
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> calcLocal = calcQueue.Get<float>();
    LocalTensor<float> resultLocal = resultQueue.Get<float>();
    int32_t num = static_cast<int32_t>(currentNum);

    if constexpr (sizeof(T) == sizeof(float)) {
        Adds(calcLocal, xLocal, 0.0f, num);
    } else {
        Cast(calcLocal, xLocal, RoundMode::CAST_NONE, num);
    }
    Mins(calcLocal, calcLocal, EXP_LIMIT, num);
    Maxs(calcLocal, calcLocal, -EXP_LIMIT, num);
    Muls(calcLocal, calcLocal, -1.0f, num);
    Exp(calcLocal, calcLocal, num);
    Adds(calcLocal, calcLocal, 1.0f, num);
    Duplicate(resultLocal, 1.0f, num);
    Div(resultLocal, resultLocal, calcLocal, num);
    if constexpr (sizeof(T) == sizeof(float)) {
        Adds(yLocal, resultLocal, 0.0f, num);
    } else {
        Cast(yLocal, resultLocal, RoundMode::CAST_NONE, num);
    }
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
{
    // TODO: 实现 Process 逻辑
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
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