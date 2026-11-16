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

#include <cstdint>

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t DMA_ALIGNMENT = 32;
constexpr uint32_t MAX_TILE_BYTES = 32 * 1024;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count);
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count);
    __aicore__ inline void Compute(uint32_t count);
    __aicore__ inline uint32_t CountAt(uint64_t offset) const;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, 1> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    uint64_t blockLength_ = 0;
    uint32_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    if (tilingData->totalNum <= 0 || tilingData->blockFactor <= 0 ||
        tilingData->ubFactor <= 0) {
        return;
    }

    const uint64_t totalNum = static_cast<uint64_t>(tilingData->totalNum);
    const uint64_t blockFactor = static_cast<uint64_t>(tilingData->blockFactor);
    const uint64_t blockIndex = AscendC::GetBlockIdx();
    if (blockIndex > (totalNum - 1) / blockFactor) {
        return;
    }

    const uint64_t start = blockIndex * blockFactor;
    const uint64_t remaining = totalNum - start;
    blockLength_ = remaining < blockFactor ? remaining : blockFactor;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + start, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + start, blockLength_);

    constexpr uint32_t maxTileElements = MAX_TILE_BYTES / sizeof(T);
    const auto ubFactor = static_cast<uint64_t>(tilingData->ubFactor);
    ubLength_ = ubFactor < maxTileElements ? static_cast<uint32_t>(ubFactor) : maxTileElements;
    const bool singleTile = blockLength_ <= ubLength_;
    const uint8_t bufferCount = singleTile ? 1 : BUFFER_NUM;
    const uint32_t bufferElements = singleTile ? static_cast<uint32_t>(blockLength_) : ubLength_;
    const uint32_t tileBytes = bufferElements * sizeof(T);
    const uint32_t bufferBytes = (tileBytes + DMA_ALIGNMENT - 1) / DMA_ALIGNMENT * DMA_ALIGNMENT;
    pipe.InitBuffer(inputQueueX, bufferCount, bufferBytes);
    pipe.InitBuffer(outputQueueY, bufferCount, bufferBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(uint64_t offset, uint32_t count)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const uint32_t bytes = count * sizeof(T);
    if (bytes % DMA_ALIGNMENT == 0) {
        DataCopy(xLocal, inputGMX[offset], count);
    } else {
        const DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        const DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(uint32_t count)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(count));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(uint64_t offset, uint32_t count)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const uint32_t bytes = count * sizeof(T);
    if (bytes % DMA_ALIGNMENT == 0) {
        DataCopy(outputGMY[offset], yLocal, count);
    } else {
        const DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline uint32_t Relu<T>::CountAt(uint64_t offset) const
{
    const uint64_t remaining = blockLength_ - offset;
    return remaining < ubLength_ ? static_cast<uint32_t>(remaining) : ubLength_;
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }

    uint64_t offset = 0;
    uint32_t count = CountAt(offset);
    CopyIn(offset, count);
    while (offset + count < blockLength_) {
        const uint64_t nextOffset = offset + count;
        const uint32_t nextCount = CountAt(nextOffset);
        // 先为下一块发起搬入，再计算并写回当前块。
        CopyIn(nextOffset, nextCount);
        Compute(count);
        CopyOut(offset, count);
        offset = nextOffset;
        count = nextCount;
    }
    Compute(count);
    CopyOut(offset, count);
}

} // namespace NsRelu
#endif // RELU_H
