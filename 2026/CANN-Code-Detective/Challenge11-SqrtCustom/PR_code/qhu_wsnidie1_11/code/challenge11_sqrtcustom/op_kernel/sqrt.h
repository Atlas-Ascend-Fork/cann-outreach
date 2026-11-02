/*!
 * \file sqrt.h
 * \brief Sqrt 算子 kernel 类定义
 */

#ifndef SQRT_H
#define SQRT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sqrt_tiling_data.h"
#include "sqrt_tiling_key.h"

namespace NsSqrt {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t UB_SIZE_BYTES = 192 * 1024; // 单核 UB 容量

template <typename T>
class Sqrt {
public:
    __aicore__ inline Sqrt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData);
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
__aicore__ inline void Sqrt<T>::Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
{
    // 每核处理的数据长度（末尾核按 totalNum 截断）
    int64_t blockOffset = static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->blockFactor;
    this->blockLength_ = tilingData->blockFactor;
    if (blockOffset >= tilingData->totalNum) {
        this->blockLength_ = 0;
    } else if (blockOffset + this->blockLength_ > tilingData->totalNum) {
        this->blockLength_ = tilingData->totalNum - blockOffset;
    }
    // UB 内分块长度按 UB 容量约束（输入/输出各 BUFFER_NUM 块）做自适应
    int64_t ubLimit = UB_SIZE_BYTES / (2 * BUFFER_NUM * static_cast<int64_t>(sizeof(T)));
    ubLimit = ubLimit / 64 * 64;
    this->ubLength_ = tilingData->ubFactor < ubLimit ? tilingData->ubFactor : ubLimit;
    if (this->ubLength_ <= 0) {
        this->ubLength_ = 64;
    }

    // 设置每个核的 Global Memory 起始地址（多核切分）
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, this->blockLength_);

    // 为队列分配 UB 内存（双缓冲）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Sqrt(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Process()
{
    int64_t loopCount = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t remaining = this->blockLength_ - i * this->ubLength_;
        int64_t currentNum = remaining < this->ubLength_ ? remaining : this->ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsSqrt
#endif // SQRT_H
