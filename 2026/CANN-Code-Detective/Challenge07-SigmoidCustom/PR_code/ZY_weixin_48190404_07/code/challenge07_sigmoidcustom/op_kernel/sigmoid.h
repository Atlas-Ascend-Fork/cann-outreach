/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Sigmoid {
public:
    __aicore__ inline Sigmoid(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData);
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
__aicore__ inline void Sigmoid<T>::Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
     // 获取当前核的索引及 tiling 参数
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t blockFactor = tilingData->blockFactor;
    int64_t totalNum = tilingData->totalNum;

    // 计算当前核处理的全局内存起始偏移及长度
    int64_t offset = blockIdx * blockFactor;
    blockLength_ = (offset + blockFactor > totalNum) ? (totalNum - offset) : blockFactor;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    // 设置全局张量的偏移和长度
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    // 获取 UB 切分长度（如果 tiling 未正确设置，则回退到整个 blockLength）
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0 || ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    // 初始化队列缓冲区
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    AscendC::LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(inputLocal, inputGMX[progress], currentNum);
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    AscendC::LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();

    // 计算 y = 1 / (1 + exp(-x))
    // 1. x = -x
    AscendC::Muls(inputLocal, inputLocal, static_cast<T>(-1.0), currentNum);
    // 2. x = exp(-x)
    AscendC::Exp(inputLocal, inputLocal, currentNum);
    // 3. x = 1 + exp(-x)
    AscendC::Adds(inputLocal, inputLocal, static_cast<T>(1.0), currentNum);
    // 4. y = 1
    AscendC::Duplicate(outputLocal, static_cast<T>(1.0), currentNum);
    // 5. y = 1 / (1 + exp(-x))
    AscendC::Div(outputLocal, outputLocal, inputLocal, currentNum);

    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    AscendC::LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopy(outputGMY[progress], outputLocal, currentNum);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
{
    // TODO: 实现 Process 逻辑
    if (blockLength_ == 0) {
        return;
    }

    // 计算当前核需要循环处理的次数
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        // 计算最后一次循环的实际处理数量
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
        CopyIn(i * ubLength_, currentNum);
        Compute(currentNum);
        CopyOut(i * ubLength_, currentNum);
    }
}

} // namespace NsSigmoid
#endif // SIGMOID_H
