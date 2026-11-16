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
    int64_t processLength_ = 0; // 记录当前核实际处理长度
};

// ================= 实现具体的 kernel 逻辑 =================

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    this->blockLength_ = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    int64_t startIdx = blockIdx * this->blockLength_;
    int64_t remainder = tilingData->totalNum - startIdx;

    if (remainder < 0) remainder = 0;
    this->processLength_ = (remainder < this->blockLength_) ? remainder : this->blockLength_;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startIdx, this->processLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startIdx, this->processLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    // 严格使用 DataCopyPad 防止任何越界和对齐错误
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * this->ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // ReLU 核心逻辑：y = max(0, x)
    AscendC::Maxs(yLocal, xLocal, (T)0.0, currentNum);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * this->ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (this->processLength_ == 0) return;

    int64_t loopCount = (this->processLength_ + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = this->ubLength_;
        if (i == loopCount - 1) {
            currentNum = this->processLength_ - i * this->ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H