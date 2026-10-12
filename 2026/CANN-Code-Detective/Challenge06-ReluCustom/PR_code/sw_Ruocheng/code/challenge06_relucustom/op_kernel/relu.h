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

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    const int64_t totalNum = tilingData->totalNum;
    const int64_t blockFactor = (tilingData->blockFactor > 0) ? tilingData->blockFactor : totalNum;
    ubLength_ = (tilingData->ubFactor > 0) ? tilingData->ubFactor : blockFactor;

    // 每个核的起始偏移；末尾核可能不足 blockFactor 个元素，按剩余长度截断避免越界
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t startIdx = blockIdx * blockFactor;
    blockLength_ = 0;
    if (startIdx < totalNum) {
        const int64_t remainNum = totalNum - startIdx;
        blockLength_ = (remainNum < blockFactor) ? remainNum : blockFactor;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + startIdx);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + startIdx);

    // in / out 各 BUFFER_NUM 份 UB buffer，共 2 * BUFFER_NUM * ubLength_ * sizeof(T) 字节
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T))));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // 类名也为 Relu，必须显式限定命名空间，否则会解析成注入类名 NsRelu::Relu
    AscendC::Relu(yLocal, xLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    const int64_t loopCount = blockLength_ / ubLength_;
    const int64_t tailNum = blockLength_ % ubLength_;

    for (int64_t i = 0; i < loopCount; i++) {
        const int64_t progress = i * ubLength_;
        CopyIn(progress, ubLength_);
        Compute(ubLength_);
        CopyOut(progress, ubLength_);
    }
    if (tailNum > 0) {
        const int64_t progress = loopCount * ubLength_;
        CopyIn(progress, tailNum);
        Compute(tailNum);
        CopyOut(progress, tailNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
