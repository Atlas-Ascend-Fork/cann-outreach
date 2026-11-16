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

// DoubleBuffer：输入、输出队列各开 2 块 UB，实现 MTE2/V/MTE3 流水
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
    // 本核在全局数据中的起始偏移：blockFactor 由 host 侧 tiling 按核数均分得到
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t blockFactor = tilingData->blockFactor;
    int64_t offset = blockIdx * blockFactor;
    if (offset > tilingData->totalNum) {
        offset = tilingData->totalNum;
    }
    // 尾核实际处理的数据量可能不足 blockFactor
    int64_t remain = tilingData->totalNum - offset;
    blockLength_ = (remain < blockFactor) ? remain : blockFactor;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        // 防御：至少保证一个 32B 数据块，避免 UB 申请/循环异常
        ubLength_ = static_cast<int64_t>(32 / sizeof(T));
    }

    // 设置每个核的 Global Memory 起始地址（多核切分的关键）
    uint32_t gmLength = static_cast<uint32_t>(blockLength_);
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, gmLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, gmLength);

    // 为队列分配 UB 内存，每块 ubLength_ 个元素，各 BUFFER_NUM 块
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
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
    // y = max(0, x)，与 torch.nn.functional.relu 语义一致
    AscendC::Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));
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
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = blockLength_ - i * ubLength_;
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