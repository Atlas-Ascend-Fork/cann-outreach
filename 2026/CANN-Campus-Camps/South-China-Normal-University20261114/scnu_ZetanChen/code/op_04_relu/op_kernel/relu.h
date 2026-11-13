/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_TILE_BYTES = 16 * 1024;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
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
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    const int64_t offset =
        static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    if (remaining <= 0 || tilingData->blockFactor <= 0 || tilingData->ubFactor <= 0) {
        return;
    }
    blockLength_ = remaining < tilingData->blockFactor ?
        remaining : tilingData->blockFactor;
    const int64_t maxTileElements = MAX_TILE_BYTES / sizeof(T);
    ubLength_ = tilingData->ubFactor < maxTileElements ?
        tilingData->ubFactor : maxTileElements;
    const uint32_t bufferBytes = static_cast<uint32_t>(
        (ubLength_ * sizeof(T) + 31) / 32 * 32);

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const uint32_t count = static_cast<uint32_t>(currentNum);
    if ((currentNum * sizeof(T)) % 32 == 0) {
        DataCopy(xLocal, inputGMX[progress * ubLength_], count);
    } else {
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const uint32_t count = static_cast<uint32_t>(currentNum);
    if ((currentNum * sizeof(T)) % 32 == 0) {
        DataCopy(outputGMY[progress * ubLength_], yLocal, count);
    } else {
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    CopyIn(0, blockLength_ < ubLength_ ? blockLength_ : ubLength_);
    for (int64_t i = 0; i < loopCount; ++i) {
        // 使用第二个输入 buffer 预取下一块。
        if (i + 1 < loopCount) {
            const int64_t nextRemaining = blockLength_ - (i + 1) * ubLength_;
            CopyIn(i + 1, nextRemaining < ubLength_ ? nextRemaining : ubLength_);
        }
        const int64_t remaining = blockLength_ - i * ubLength_;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
