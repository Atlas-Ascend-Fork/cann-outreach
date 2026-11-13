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

constexpr int32_t BUFFER_NUM = 2;   // 每队列 buffer 块数（双缓冲）

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

// 每个核负责 [start, start + blockLength_)，start = blockIdx * blockFactor
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    if (blockFactor <= 0) {
        blockFactor = totalNum;
    }

    // 本核的起始偏移与实际负责长度（最后一个核可能不足 blockFactor）
    int64_t start = static_cast<int64_t>(AscendC::GetBlockIdx()) * blockFactor;
    int64_t blockLength = totalNum - start;
    if (blockLength > blockFactor) {
        blockLength = blockFactor;
    }
    if (blockLength < 0) {
        blockLength = 0;
    }

    // 单次 UB 循环长度：不超过本核长度，并按 32B 对齐（DataCopy 要求）
    int64_t alignNum = 32 / static_cast<int64_t>(sizeof(T));
    int64_t ubLength = (tilingData->ubFactor > 0) ? tilingData->ubFactor : blockLength;
    if (blockLength > 0 && ubLength > blockLength) {
        ubLength = blockLength;
    }
    ubLength = (ubLength / alignNum) * alignNum;
    if (ubLength <= 0) {
        ubLength = alignNum;   // 空核也要给出合法的 buffer 大小
    }

    this->blockLength_ = blockLength;
    this->ubLength_ = ubLength;

    // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, blockLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, blockLength);

    // 为队列分配 UB 内存：BUFFER_NUM 块即可实现双缓冲
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // y = max(0, x)
    AscendC::Relu(yLocal, xLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (this->blockLength_ <= 0) {
        return;   // 尾部的空核直接退出
    }
    // 按 ubLength_ 分块推进，最后一块按剩余长度裁剪
    int64_t loopCount = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = this->blockLength_ - i * this->ubLength_;
        if (currentNum > this->ubLength_) {
            currentNum = this->ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H