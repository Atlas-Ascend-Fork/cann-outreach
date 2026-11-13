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
    negativeSlope_ = tilingData->negativeSlope;
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (offset >= tilingData->totalNum || tilingData->ubFactor <= 0) {
        return;
    }
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    const uint8_t bufferNum = blockLength_ <= ubLength_ ? 1 : BUFFER_NUM;
    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX, bufferNum, bufferBytes);
    pipe.InitBuffer(outputQueueY, bufferNum, bufferBytes);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    constexpr int64_t alignNum = 32 / sizeof(T);
    if (currentNum % alignNum == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    const int32_t count = static_cast<int32_t>(currentNum);
    const T slope = static_cast<T>(negativeSlope_);
    AscendC::Maxs(yLocal, xLocal, static_cast<T>(0), count);
    AscendC::Mins(xLocal, xLocal, static_cast<T>(0), count);
    AscendC::Muls(xLocal, xLocal, slope, count);
    AscendC::Add(yLocal, yLocal, xLocal, count);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    constexpr int64_t alignNum = 32 / sizeof(T);
    if (currentNum % alignNum == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }
    if (blockLength_ <= ubLength_) {
        CopyIn(0, blockLength_);
        Compute(blockLength_);
        CopyOut(0, blockLength_);
        return;
    }
    const int64_t fullTiles = blockLength_ / ubLength_;
    const int64_t tailNum = blockLength_ % ubLength_;
    for (int64_t i = 0; i < fullTiles; ++i) {
        CopyIn(i, ubLength_);
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }
    if (tailNum != 0) {
        CopyIn(fullTiles, tailNum);
        Compute(tailNum);
        CopyOut(fullTiles, tailNum);
    }
}

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
