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

    int64_t totalNum_ = 0;    // 总元素数量
    int64_t blockLength_ = 0; // 每个核处理的元素数量
    int64_t ubLength_ = 0;    // 每次 UB 循环处理的元素数量
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    // 防御性处理，保证后续循环和 UB 申请不会出现 0
    if (blockLength_ <= 0) {
        blockLength_ = 1;
    }
    if (ubLength_ <= 0) {
        ubLength_ = blockLength_;
    }

    // 设置每个核的 Global Memory 起始地址（第 blockIdx 个核从 blockIdx * blockLength_ 开始）
    const int64_t blockIdx = GetBlockIdx();
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockIdx * blockLength_, static_cast<uint32_t>(blockLength_));
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockIdx * blockLength_, static_cast<uint32_t>(blockLength_));

    // 输入/输出队列各申请 BUFFER_NUM(2) 块，即 DoubleBuffer
    const uint32_t ubBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    const int32_t calCount = static_cast<int32_t>(currentNum);

    // y = max(0, x)
    // 注意：类名也叫 Relu，这里必须用 AscendC::Relu 限定，否则会被解析成构造函数
    AscendC::Relu(yLocal, xLocal, calCount);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (totalNum_ <= 0) {
        return;
    }

    // 按核切分：第 blockIdx 个核处理 [blockIdx * blockLength_, (blockIdx + 1) * blockLength_)
    const int64_t blockIdx = GetBlockIdx();
    const int64_t offset = blockIdx * blockLength_;
    if (offset >= totalNum_) {
        return;
    }

    int64_t num = totalNum_ - offset;
    if (num > blockLength_) {
        num = blockLength_;
    }

    // UB 内按 ubLength_ 分块循环（blockFactor/ubFactor 均按 32B 对齐）
    const int64_t loopCount = (num + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = num - i * ubLength_;
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