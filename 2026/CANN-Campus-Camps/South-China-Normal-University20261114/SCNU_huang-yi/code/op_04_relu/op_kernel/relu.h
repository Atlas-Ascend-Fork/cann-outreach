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
    // 每个核处理的元素个数
    blockLength_ = tilingData->blockFactor;
    // 每次 UB 循环处理的元素个数
    ubLength_ = tilingData->ubFactor;

    // 多核切分：按 blockIdx 偏移 Global Memory 起始地址，每个核只搬/写自己那段
    int64_t blockIdx = GetBlockIdx();
    int64_t offset = blockLength_ * blockIdx;

    // 防止最后一个核越界（totalNum 能被 blockDim 整除时不会触发，这里做兜底）
    int64_t actualLength = blockLength_;
    if (offset + actualLength > tilingData->totalNum) {
        actualLength = (tilingData->totalNum > offset) ? (tilingData->totalNum - offset) : 0;
    }
    blockLength_ = actualLength;

    inputGMX.SetGlobalBuffer((__gm__ T *)x + offset, actualLength);
    outputGMY.SetGlobalBuffer((__gm__ T *)y + offset, actualLength);

    // 双缓冲：每个队列分配 BUFFER_NUM 块 ubLength_ 大小的 UB 空间
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
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
    // ReLU 计算：y = max(0, x)。使用 Maxs(逐元素与标量取大)，避免依赖 unary 头文件
    AscendC::Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
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
    // 完整 tile 循环（ubFactor 为 2 的幂且整除 blockLength_，此处无尾巴）
    int64_t loopCount = blockLength_ / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        CopyIn(i, ubLength_);
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }
    // 兜底：处理最后一个可能不足 ubFactor 的尾巴
    int64_t tailLength = blockLength_ - loopCount * ubLength_;
    if (tailLength > 0) {
        CopyIn(loopCount, tailLength);
        Compute(tailLength);
        CopyOut(loopCount, tailLength);
    }
}

} // namespace NsRelu
#endif // RELU_H
