/*!
 * \file sqrt.h
 * \brief SqrtCustom 算子 kernel 类定义
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

template <typename T>
class SqrtOp {
public:
    __aicore__ inline SqrtOp(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void ProcessSingle();
    __aicore__ inline void CopyOutTail(int64_t gmOffset, const LocalTensor<T>& src, int64_t tailNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> calcBufX;
    TBuf<QuePosition::VECCALC> calcBufY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t blockLengthAligned_ = 0;  // 32B 对齐后的 blockLength_
    int64_t ubLength_ = 0;
    int64_t totalNum_ = 0;
    int64_t alignNum_ = 0;            // 32B 对齐的元素数
};

template <typename T>
__aicore__ inline void SqrtOp<T>::Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
{
    ubLength_ = tilingData->ubFactor;
    int64_t blockFactor = tilingData->blockFactor;
    totalNum_ = tilingData->totalNum;
    alignNum_ = 32 / sizeof(T);
    int64_t blockIdx = GetBlockIdx();
    // 计算本核实际处理的元素数（最后一核可能不足 blockFactor）
    int64_t startOffset = blockIdx * blockFactor;
    int64_t remainNum = totalNum_ - startOffset;
    blockLength_ = (remainNum < blockFactor) ? remainNum : blockFactor;
    if (blockLength_ <= 0) {
        return;
    }
    // 对齐长度：仅用于 UB 分配（32B 上取整）
    blockLengthAligned_ = (blockLength_ + alignNum_ - 1) / alignNum_ * alignNum_;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    if (blockLength_ <= ubLength_) {
        // 单轮场景：仅分配 TBuf，跳过 TQue 初始化减少开销
        pipe.InitBuffer(calcBufX, blockLengthAligned_ * sizeof(T));
        pipe.InitBuffer(calcBufY, blockLengthAligned_ * sizeof(T));
    } else {
        // 多轮场景：初始化双缓冲队列
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void SqrtOp<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    // DataCopyPad 的 blockLen 单位是字节（DataCopy 才是 32B 块数）
    DataCopyParams copyParams{1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void SqrtOp<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Sqrt(yLocal, xLocal, currentNum);
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void SqrtOp<T>::CopyOutTail(int64_t gmOffset, const LocalTensor<T>& src, int64_t tailNum)
{
    // 尾块逐元素写回，避免 DataCopy 非对齐越界
    for (int64_t i = 0; i < tailNum; ++i) {
        outputGMY.SetValue(gmOffset + i, src.GetValue(i));
    }
}

template <typename T>
__aicore__ inline void SqrtOp<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    int64_t gmOffset = progress * ubLength_;
    int64_t alignedNum = currentNum / alignNum_ * alignNum_;  // 对齐部分元素数
    int64_t tailNum = currentNum - alignedNum;                // 尾块元素数

    if (alignedNum > 0) {
        // 对齐部分用 DataCopy 写回
        DataCopy(outputGMY[gmOffset], yLocal, alignedNum);
    }
    if (tailNum > 0) {
        // 尾块逐元素写回，避免越界
        CopyOutTail(gmOffset + alignedNum, yLocal[alignedNum], tailNum);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SqrtOp<T>::ProcessSingle()
{
    // 单轮无队列路径：GM -> UB -> Sqrt -> GM，消除 TQue 同步开销
    LocalTensor<T> xLocal = calcBufX.Get<T>();
    LocalTensor<T> yLocal = calcBufY.Get<T>();

    // GM->UB：DataCopyPad 的 blockLen 单位是字节，按实际元素数读取（1 字节粒度，不越界）
    DataCopyParams copyInParams{1, static_cast<uint16_t>(blockLength_ * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX, copyInParams, padParams);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // 计算：仅处理实际元素数
    AscendC::Sqrt(yLocal, xLocal, blockLength_);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    // UB->GM：对齐部分用 DataCopy，尾块用 SetValue 逐元素写回
    int64_t alignedNum = blockLength_ / alignNum_ * alignNum_;
    int64_t tailNum = blockLength_ - alignedNum;
    if (alignedNum > 0) {
        DataCopy(outputGMY, yLocal, alignedNum);
    }
    if (tailNum > 0) {
        CopyOutTail(alignedNum, yLocal[alignedNum], tailNum);
    }
}

template <typename T>
__aicore__ inline void SqrtOp<T>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    if (loopCount == 1) {
        ProcessSingle();
    } else {
        for (int64_t i = 0; i < loopCount; ++i) {
            int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }
}

} // namespace NsSqrt
#endif // SQRT_H
