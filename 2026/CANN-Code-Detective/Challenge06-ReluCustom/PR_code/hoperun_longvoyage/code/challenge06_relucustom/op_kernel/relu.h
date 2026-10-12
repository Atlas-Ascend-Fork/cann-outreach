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
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 多核切分：核 i 处理 [i*blockFactor, min((i+1)*blockFactor, totalNum))
    // 最后一核可能不满 blockFactor（totalNum 不整除 blockNum 时），用 totalNum 收尾
    int64_t start = tilingData->blockFactor * AscendC::GetBlockIdx();
    int64_t remain = tilingData->totalNum - start;
    this->blockLength_ = remain > 0 ? (remain < tilingData->blockFactor ? remain : tilingData->blockFactor) : 0;
    this->ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, this->blockLength_);
    // 为队列分配 UB 内存
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // relu(x) = max(0, x)
    AscendC::Maxs(yLocal, xLocal, (T)0.0f, currentNum);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // 单核内按 ubFactor 分块循环，最后一块处理剩余元素
    int64_t loopCount = this->blockLength_ / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        CopyIn(i, this->ubLength_);
        Compute(this->ubLength_);
        CopyOut(i, this->ubLength_);
    }
    int64_t remain = this->blockLength_ % this->ubLength_;
    if (remain > 0) {
        CopyIn(loopCount, remain);
        Compute(remain);
        CopyOut(loopCount, remain);
    }
}

} // namespace NsRelu
#endif // RELU_H
