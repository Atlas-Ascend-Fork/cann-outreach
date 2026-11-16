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

constexpr int32_t BUFFER_NUM = 2; // 双缓冲流水：搬运与计算重叠

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

    int64_t blockLength_ = 0;  // 本核负责的元素数
    int64_t ubLength_ = 0;     // 单次 UB 循环处理的元素数
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 按核号切分全局数据，尾核处理剩余部分
    int64_t offset = AscendC::GetBlockIdx() * tilingData->blockFactor;
    int64_t currentBlockLength = tilingData->totalNum - offset;
    if (currentBlockLength > tilingData->blockFactor) {
        currentBlockLength = tilingData->blockFactor;
    }
    if (currentBlockLength < 0) {
        currentBlockLength = 0;
    }
    this->blockLength_ = currentBlockLength;
    this->ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, this->blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
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
    AscendC::Relu(yLocal, xLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
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
    int64_t progress = 0;
    int64_t currentNum = this->blockLength_;
    if (currentNum > this->ubLength_) {
        currentNum = this->ubLength_;
    }
    CopyIn(progress, currentNum);
    progress += currentNum;
    while (progress < this->blockLength_) {
        int64_t nextNum = this->blockLength_ - progress;
        if (nextNum > this->ubLength_) {
            nextNum = this->ubLength_;
        }
        CopyIn(progress, nextNum);
        Compute(currentNum);
        CopyOut(progress - currentNum, currentNum);
        progress += nextNum;
        currentNum = nextNum;
    }
    Compute(currentNum);
    CopyOut(progress - currentNum, currentNum);
}

} // namespace NsRelu
#endif // RELU_H
