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

constexpr int32_t BUFFER_NUM = 2;   // DoubleBuffer

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

    int64_t alignElemNum_ = 1;  // 32B 对齐的元素个数
    int64_t blockOffset_ = 0;   // 当前核在 GM 上的起始元素偏移
    int64_t blockLength_ = 0;   // 当前核处理的元素总数
    int64_t ubLength_ = 0;      // 每次 UB 循环处理的元素数
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    alignElemNum_ = 32 / static_cast<int64_t>(sizeof(T));

    int64_t blockFactor = tilingData->blockFactor;
    blockOffset_ = GetBlockIdx() * blockFactor;
    blockLength_ = tilingData->totalNum - blockOffset_;
    if (blockLength_ > blockFactor) {
        blockLength_ = blockFactor;
    }
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    // 每个核按 blockFactor 划分 GM 起始地址（blockFactor 已按 32B 对齐，偏移天然对齐）
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockOffset_, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockOffset_, blockLength_);

    int64_t ubFactor = tilingData->ubFactor;
    ubLength_ = ubFactor < blockLength_ ? ubFactor : blockLength_;
    if (ubLength_ <= 0) {
        return;  // 无数据的核，Process 直接返回
    }

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    int64_t offset = progress * ubLength_;
    if (currentNum % alignElemNum_ == 0) {
        // 32B 对齐时走 DMA 快速路径
        DataCopy(xLocal, inputGMX[offset], currentNum);
    } else {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // y = max(0, x)
    AscendC::Relu(yLocal, xLocal, currentNum);
    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    int64_t offset = progress * ubLength_;
    if (currentNum % alignElemNum_ == 0) {
        DataCopy(outputGMY[offset], yLocal, currentNum);
    } else {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H