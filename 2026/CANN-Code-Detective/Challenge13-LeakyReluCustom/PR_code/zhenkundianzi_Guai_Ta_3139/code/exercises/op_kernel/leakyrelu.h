/*!
 * \file leakyrelu.h
 * \brief Leakyrelu 算子 kernel 类定义
 */

#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "leakyrelu_tiling_data.h"
#include "leakyrelu_tiling_key.h"

namespace NsLeakyrelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Leakyrelu {
public:
    __aicore__ inline Leakyrelu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData);
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

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float negativeSlope_ = 0.01f;
};

template <typename T>
__aicore__ inline void Leakyrelu<T>::Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
{
    uint32_t blockIdx = GetBlockIdx();
    int64_t offset = static_cast<int64_t>(blockIdx) * tilingData->blockFactor;

    blockLength_ = tilingData->blockFactor;
    if (offset + blockLength_ > tilingData->totalNum) {
        blockLength_ = tilingData->totalNum - offset;
        if (blockLength_ < 0) {
            blockLength_ = 0;
        }
    }

    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }
    negativeSlope_ = tilingData->negativeSlope;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    // InitBuffer 的 bufSize 需要 32B 对齐
    uint32_t bufSize = static_cast<uint32_t>(ubLength_ * sizeof(T));
    bufSize = (bufSize + 31U) & ~31U;
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufSize);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufSize);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    int64_t offset = progress * ubLength_;
    constexpr int64_t alignNum = 32 / sizeof(T);
    if (currentNum % alignNum == 0) {
        DataCopy(inputLocal, inputGMX[offset], static_cast<int32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();

    // y = max(x, 0) + negative_slope * min(x, 0)
    Maxs(outputLocal, inputLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));
    Mins(inputLocal, inputLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));
    Muls(inputLocal, inputLocal, static_cast<T>(negativeSlope_), static_cast<int32_t>(currentNum));
    Add(outputLocal, outputLocal, inputLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    int64_t offset = progress * ubLength_;
    constexpr int64_t alignNum = 32 / sizeof(T);
    if (currentNum % alignNum == 0) {
        DataCopy(outputGMY[offset], outputLocal, static_cast<int32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], outputLocal, copyParams);
    }
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
