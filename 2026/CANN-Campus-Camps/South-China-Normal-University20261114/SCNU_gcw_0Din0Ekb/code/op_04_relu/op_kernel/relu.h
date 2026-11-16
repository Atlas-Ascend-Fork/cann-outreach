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
constexpr int64_t UB_BLOCK_BYTES = 32; // UB 最小访问粒度（字节），buffer 按此对齐分配

__aicore__ inline int64_t ReluCeilDiv(int64_t value, int64_t factor)
{
    return (value + factor - 1) / factor;
}

__aicore__ inline int64_t ReluAlignUp(int64_t value, int64_t align)
{
    return ReluCeilDiv(value, align) * align;
}

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
    __aicore__ inline int64_t TileLen(int64_t progress); // 第 progress 块实际元素数（尾块会短）

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
    // 每个核绑定到自己负责的那段 Global Memory，起始地址 = 核索引 * 每核元素数
    int64_t startOffset = static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->blockFactor;

    // 尾核实际数据量可能不足 blockFactor，按 totalNum 截断，避免读出/写出越界
    blockLength_ = tilingData->totalNum - startOffset;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    if (blockLength_ > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    }

    // 单次 UB 处理量不超过本核的数据量
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }
    if (ubLength_ <= 0) {
        ubLength_ = 1; // 空 tensor 时仅用于避免除零，Process 会直接返回
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    // UB 按 32B 对齐分配（末尾不足一个块的部分留白，不参与搬入搬出）
    int64_t alignElem = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    int64_t alignedUbLength = ReluAlignUp(ubLength_, alignElem);
    uint32_t queueBytes = static_cast<uint32_t>(alignedUbLength * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, queueBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, queueBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    int64_t alignElem = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    if (currentNum % alignElem == 0) {
        // 整块对齐：走轻量的 DataCopy
        DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    } else {
        // 按字节描述单行连续数据，允许尾部元素不足一个 32B 块
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // 矢量指令按 32B 对齐计算，尾块不足一个块的部分在 buffer 留白区内完成，CopyOut 只回拷有效元素
    int64_t alignElem = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    uint32_t calcNum = static_cast<uint32_t>(ReluAlignUp(currentNum, alignElem));
    // y = max(0, x)：一条矢量指令完成，Maxs 对 fp16 / fp32 均支持
    Maxs(yLocal, xLocal, static_cast<T>(0), calcNum);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    int64_t alignElem = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    if (currentNum % alignElem == 0) {
        DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline int64_t Relu<T>::TileLen(int64_t progress)
{
    int64_t currentNum = blockLength_ - progress * ubLength_;
    if (currentNum > ubLength_) {
        currentNum = ubLength_;
    }
    return (currentNum > 0) ? currentNum : 0;
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // 本核没有数据（空 tensor 或多余核）时直接返回
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = ReluCeilDiv(blockLength_, ubLength_);
    if (loopCount <= 0) {
        return;
    }

    // 三阶段软件流水（DoubleBuffer + 预取）：
    // 先把第 0 块搬入，之后每轮在“算当前块”之前先把“下一块”的搬入发出去，
    // 这样 MTE2（GM->UB）、V（矢量）、MTE3（UB->GM）三条流水同时都有指令在跑，
    // 不会出现“算的时候搬运单元空转、搬的时候计算单元空转”的流水气泡。
    CopyIn(0, TileLen(0));
    for (int64_t i = 0; i < loopCount; i++) {
        if (i + 1 < loopCount) {
            CopyIn(i + 1, TileLen(i + 1)); // 预取下一块，与本块计算/搬出重叠
        }
        Compute(TileLen(i));
        CopyOut(i, TileLen(i));
    }
}

} // namespace NsRelu
#endif // RELU_H
