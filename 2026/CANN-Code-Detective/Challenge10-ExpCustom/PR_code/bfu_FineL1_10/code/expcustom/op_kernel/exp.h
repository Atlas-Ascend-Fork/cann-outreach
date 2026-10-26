/*!
 * \file exp.h
 * \brief Exp 算子 kernel 类定义
 */

#ifndef EXP_H
#define EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Exp {
public:
    __aicore__ inline Exp(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData);
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

// 初始化当前 block 的输入输出范围，并按 ubLength_ 配置队列缓存。
template <typename T>
__aicore__ inline void Exp<T>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
{
    const int64_t totalNum = tilingData->totalNum;
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * blockFactor;

    if (totalNum <= 0 || blockFactor <= 0 || blockOffset >= totalNum) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }

    blockLength_ = totalNum - blockOffset;
    if (blockLength_ > blockFactor) {
        blockLength_ = blockFactor;
    }

    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0 || ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);

    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
}

// 将当前 tile 从 GM 搬运到输入队列。
template <typename T>
__aicore__ inline void Exp<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue<T>(xLocal);
}

// 逐元素计算 y = exp(x)。
template <typename T>
__aicore__ inline void Exp<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    AscendC::Exp(yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// 将计算结果从输出队列写回 GM。
template <typename T>
__aicore__ inline void Exp<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

// 每个 block 循环处理自己的数据段，最后一个 tile 按剩余元素数处理。
template <typename T>
__aicore__ inline void Exp<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    int64_t progress = 0;
    while (progress < blockLength_) {
        int64_t currentNum = blockLength_ - progress;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);

        progress += currentNum;
    }
}

} // namespace NsExp
#endif // EXP_H
