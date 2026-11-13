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

constexpr int32_t BUFFER_NUM = 2;          // 双缓冲，搬运与计算流水
constexpr uint32_t UB_BUDGET = 160 * 1024; // 单核 UB 预算（预留余量）
constexpr uint32_t ALIGN_BYTES = 32;       // DataCopy 要求 32B 对齐

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

    int64_t blockLength_ = 0; // 本核处理的元素数
    int64_t ubLength_ = 0;    // 单次搬运 / 计算的元素数
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    constexpr uint32_t alignNum = static_cast<uint32_t>(ALIGN_BYTES / sizeof(T));
    const int64_t totalLength = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0 || ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    // 按 UB 预算收缩单次处理量：输入/输出各 BUFFER_NUM 块，并保证 32B 对齐
    constexpr uint32_t perElementBytes = sizeof(T) * BUFFER_NUM * 2;
    uint32_t maxUbLength = UB_BUDGET / perElementBytes;
    maxUbLength = maxUbLength / alignNum * alignNum;
    if (maxUbLength < alignNum) {
        maxUbLength = alignNum;
    }
    if (ubLength_ > static_cast<int64_t>(maxUbLength)) {
        ubLength_ = static_cast<int64_t>(maxUbLength);
    }

    // 本核在 GM 上的起始偏移，末核长度不足时按实际长度截断
    int64_t offset = blockLength_ * static_cast<int64_t>(AscendC::GetBlockIdx());
    if (offset >= totalLength) {
        blockLength_ = 0;
        offset = totalLength;
    } else if (offset + blockLength_ > totalLength) {
        blockLength_ = totalLength - offset;
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    // 为输入/输出队列分配 UB 内存（双缓冲）
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
    // y = max(0, x)
    Maxs(yLocal, xLocal, static_cast<T>(0.0f), static_cast<int32_t>(currentNum));
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
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = ubLength_;
        if (i == loopCount - 1) {
            currentNum = blockLength_ - i * ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
