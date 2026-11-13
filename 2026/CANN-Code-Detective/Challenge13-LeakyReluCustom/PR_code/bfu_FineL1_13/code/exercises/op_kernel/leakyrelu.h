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
    float negativeSlope_ = 0.01F;
};

// 初始化GM地址、当前核负责的数据范围，以及输入/输出双缓冲队列。
template <typename T>
__aicore__ inline void Leakyrelu<T>::Init(
    GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
{
    const int64_t totalNum = tilingData->totalNum;
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * blockFactor;

    blockLength_ = totalNum - blockOffset;
    if (blockLength_ > blockFactor) {
        blockLength_ = blockFactor;
    }
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    ubLength_ = tilingData->ubFactor;
    if (ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }
    negativeSlope_ = tilingData->negativeSlope;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);

    if (ubLength_ > 0) {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    }
}

// 将当前Tile从Global Memory搬运到VECIN队列。
template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    DataCopy(inputLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(inputLocal);
}

// 使用Ascend C LeakyRelu矢量接口完成逐元素计算。
template <typename T>
__aicore__ inline void Leakyrelu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();

    LeakyRelu(
        outputLocal,
        inputLocal,
        static_cast<T>(negativeSlope_),
        static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

// 将计算结果从VECOUT队列搬运回Global Memory。
template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], outputLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(outputLocal);
}

// 每个核独立处理自己的数据段；当数据段大于UB Tile时循环分块处理。
template <typename T>
__aicore__ inline void Leakyrelu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t progress = 0; progress < loopCount; ++progress) {
        const int64_t offset = progress * ubLength_;
        int64_t currentNum = blockLength_ - offset;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
