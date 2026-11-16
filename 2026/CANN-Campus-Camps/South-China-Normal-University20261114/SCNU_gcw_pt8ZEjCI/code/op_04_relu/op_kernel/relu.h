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

    int64_t blockOffset_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t alignNum_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    alignNum_ = 32 / sizeof(T);
    int64_t blockIdx = GetBlockIdx();
    blockOffset_ = blockIdx * tilingData->blockFactor;

    int64_t remainNum = tilingData->totalNum - blockOffset_;
    blockLength_ = (remainNum > tilingData->blockFactor)
        ? tilingData->blockFactor : remainNum;
    if (blockLength_ < 0) blockLength_ = 0;

    ubLength_ = tilingData->ubFactor;

    // 直接将基址偏移融入 GlobalTensor，减少后续指令运算
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset_, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset_, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    // 4KB/32B 对齐的数据走高效的 DataCopy 指令
    if (currentNum % alignNum_ == 0) {
        DataCopy(xLocal, inputGMX[progress], currentNum);
    } else {
        // 只有纯尾部对齐破坏的块才使用 DataCopyPad
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0
        };
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
    }

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // 使用最底层的 Maxs 替代 Relu 算子，等价于 max(x, 0)，单条汇编指令
    Maxs(yLocal, xLocal, static_cast<T>(0), currentNum);

    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    if (currentNum % alignNum_ == 0) {
        DataCopy(outputGMY[progress], yLocal, currentNum);
    } else {
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0
        };
        DataCopyPad(outputGMY[progress], yLocal, copyParams);
    }

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) return;

    for (int64_t progress = 0; progress < blockLength_; progress += ubLength_) {
        int64_t currentNum = (blockLength_ - progress > ubLength_)
            ? ubLength_ : blockLength_ - progress;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu

#endif // RELU_H