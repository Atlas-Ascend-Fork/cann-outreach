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

// 单缓冲：队列深度 1，UB 全给单次搬运量（每核 5 轮 × 96KB）
// 必须与 op_host/relu_tiling.cpp 的 BUFFER_NUM_TILING 一致
constexpr int32_t BUFFER_NUM = 1;

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

    int64_t blockLength_ = 0; // 本核实际处理元素数
    int64_t ubLength_ = 0;    // 每块处理元素数
    int64_t tileNum_ = 1;     // 本核分块数（host 已算好）
};

/* ============ 每个核的 Global Memory 起始地址 ============
 * 依据：03.05 §6.5 / 08.04 §2.2.1
 * ======================================================== */
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    this->ubLength_ = tilingData->ubFactor;
    this->tileNum_ = tilingData->tileNum;

    const int64_t blockFactor = tilingData->blockFactor;
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t startOffset = blockIdx * blockFactor;

    int64_t currentLength = tilingData->totalNum - startOffset;
    if (currentLength > blockFactor) {
        currentLength = blockFactor;
    }
    if (currentLength <= 0) {
        this->blockLength_ = 0;
        return;
    }
    this->blockLength_ = currentLength;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, this->blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(this->ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(this->ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const uint32_t bytes = static_cast<uint32_t>(currentNum * sizeof(T));

    // 对齐走 DataCopy（比 DataCopyPad 快），尾块不齐才回退
    // CANN 9.0.0 的 DataCopyExtParams 为 5 参构造（第 5 位预留）
    if (bytes % 32 == 0) {
        DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
    } else {
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, inputGMX[progress * this->ubLength_], copyParams, padParams);
    }

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    // ReLU = max(0, x)：一条 Maxs 指令，fp16/fp32 走同一模板
    // 依据：03.05 §6.6 的 AscendC::Maxs(yLocal, xLocal, min, count)，min=0 即 ReLU
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    const int32_t count = static_cast<int32_t>(currentNum);

    Maxs(yLocal, xLocal, (T)0.0f, count);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const uint32_t bytes = static_cast<uint32_t>(currentNum * sizeof(T));

    if (bytes % 32 == 0) {
        DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
    } else {
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPad(outputGMY[progress * this->ubLength_], yLocal, copyParams);
    }

    outputQueueY.FreeTensor(yLocal);
}

/* 队列深度 1：必须串行三段式，不能预取（预取会与 FreeTensor 互锁） */
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // TODO: 实现 Process 逻辑
    if (this->blockLength_ <= 0 || this->ubLength_ <= 0) {
        return;
    }

    int64_t lastNum = this->blockLength_ - (this->tileNum_ - 1) * this->ubLength_;

    for (int64_t i = 0; i < this->tileNum_; i++) {
        int64_t curNum = (i == this->tileNum_ - 1) ? lastNum : this->ubLength_;
        CopyIn(i, curNum);
        Compute(curNum);
        CopyOut(i, curNum);
    }
}

} // namespace NsRelu
#endif // RELU_H