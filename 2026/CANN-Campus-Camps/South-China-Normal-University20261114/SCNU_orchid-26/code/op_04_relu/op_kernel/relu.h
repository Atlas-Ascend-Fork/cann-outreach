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

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
// 具体实现见下方各成员函数。
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (blockLength_ <= 0) {
        blockLength_ = 1;
    }
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    int64_t totalNum = tilingData->totalNum;
    inputGMX.SetGlobalBuffer((__gm__ T*)x, static_cast<uint64_t>(totalNum));
    outputGMY.SetGlobalBuffer((__gm__ T*)y, static_cast<uint64_t>(totalNum));

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    int64_t offset = GetBlockIdx() * blockLength_ + progress * ubLength_;
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    DataCopy(inputLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();

    Maxs(outputLocal, inputLocal, static_cast<T>(0), static_cast<uint32_t>(currentNum));

    inputQueueX.FreeTensor(inputLocal);
    outputQueueY.EnQue(outputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    int64_t offset = GetBlockIdx() * blockLength_ + progress * ubLength_;
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[offset], outputLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // TODO: 实现 Process 逻辑
    int64_t blockIdx = GetBlockIdx();
    int64_t totalNum = static_cast<int64_t>(inputGMX.GetSize());
    int64_t start = blockIdx * blockLength_;
    if (start >= totalNum) {
        return;
    }

    int64_t currentBlockLength = totalNum - start;
    if (currentBlockLength > blockLength_) {
        currentBlockLength = blockLength_;
    }

    int64_t ubLoop = (currentBlockLength + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < ubLoop; ++i) {
        int64_t currentNum = currentBlockLength - i * ubLength_;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H