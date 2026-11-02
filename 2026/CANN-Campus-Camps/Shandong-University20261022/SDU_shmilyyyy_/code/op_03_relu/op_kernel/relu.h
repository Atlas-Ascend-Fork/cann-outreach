/*!
 * \file relu.h
 * \brief ReLU kernel using the template's existing tiling fields.
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

template <typename T>
class Relu {
public:
    __aicore__ inline Relu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    // Queue depth is independent of the physical double-buffer count.
    TQue<QuePosition::VECIN, 1> inputQueueX;
    TQue<QuePosition::VECOUT, 1> outputQueueY;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (offset >= tilingData->totalNum || ubLength_ <= 0) {
        blockLength_ = 0;
        return;
    }
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    // Single-pass cores need one buffer; tiled cores use DoubleBuffer.
    const uint32_t bufferCount = blockLength_ <= ubLength_ ? 1u : BUFFER_NUM;
    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX, bufferCount, bufferBytes);
    pipe.InitBuffer(outputQueueY, bufferCount, bufferBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    constexpr int64_t BLOCK_ELEMS = 32 / sizeof(T);
    if (currentNum % BLOCK_ELEMS == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
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
    const int64_t offset = progress * ubLength_;
    constexpr int64_t BLOCK_ELEMS = 32 / sizeof(T);
    if (currentNum % BLOCK_ELEMS == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    // Avoid division and loop bookkeeping in the specified small-tensor case.
    if (blockLength_ <= ubLength_) {
        CopyIn(0, blockLength_);
        Compute(blockLength_);
        CopyOut(0, blockLength_);
        return;
    }

    const int64_t fullTiles = blockLength_ / ubLength_;
    for (int64_t progress = 0; progress < fullTiles; ++progress) {
        CopyIn(progress, ubLength_);
        Compute(ubLength_);
        CopyOut(progress, ubLength_);
    }
    const int64_t tailNum = blockLength_ % ubLength_;
    if (tailNum != 0) {
        CopyIn(fullTiles, tailNum);
        Compute(tailNum);
        CopyOut(fullTiles, tailNum);
    }
}

} // namespace NsRelu

#endif // RELU_H
