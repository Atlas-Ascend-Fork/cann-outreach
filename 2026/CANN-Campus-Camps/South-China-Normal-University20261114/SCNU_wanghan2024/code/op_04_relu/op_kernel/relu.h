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
    this->blockLength_ = tilingData->totalNum;
    this->ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, this->blockLength_);

    this->pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    this->pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    // 核心计算：y = max(0, x)。利用 Duplicate 生成全 0 张量，然后取最大值
    Duplicate(yLocal, (T)0.0, currentNum);
    Max(yLocal, xLocal, yLocal, currentNum);

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // 如果当前不是 0 号核，直接退出（防止多核冲突）
    if (GetBlockIdx() != 0) return; 

    // 循环处理数据块
int64_t loopCount = this->blockLength_ / this->ubLength_;
for (int64_t i = 0; i < loopCount; i++) {
    this->CopyIn(i, this->ubLength_);
    this->Compute(this->ubLength_);
    this->CopyOut(i, this->ubLength_);
}

// 处理除不尽的余数尾巴
int64_t remainder = this->blockLength_ % this->ubLength_;
if (remainder > 0) {
    int64_t currentOffset = loopCount * this->ubLength_;
    this->CopyIn(loopCount, remainder);
    this->Compute(remainder);
    this->CopyOut(loopCount, remainder);
}
}

} // namespace NsRelu
#endif // RELU_H