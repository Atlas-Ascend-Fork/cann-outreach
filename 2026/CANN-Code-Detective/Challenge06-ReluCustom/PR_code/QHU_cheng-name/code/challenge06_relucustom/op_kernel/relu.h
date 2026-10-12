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

constexpr int32_t BUFFER_NUM = 1;      // 每队列 buffer 块数（本算子按单缓冲流水）
constexpr int64_t kCopyBlockBytes = 32; // DMA 搬运的最小块粒度（字节）

__aicore__ inline int64_t LessOf(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

// 元素数按数据类型折算后是否 32B 对齐：对齐才能使用 DataCopy 整块搬运，
// 否则必须使用 DataCopyPad 按字节精确搬运，避免 GM 越界读写
template <typename T>
__aicore__ inline bool CanUsePlainCopy(int64_t elemNum)
{
    return (elemNum * static_cast<int64_t>(sizeof(T))) % kCopyBlockBytes == 0;
}

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    GlobalTensor<DT_X> inputGM;
    GlobalTensor<DT_X> outputGM;

    // 全部成员显式初始化：越界核在 Init 中提前返回时也不会残留未定义值
    int64_t blockLength = 0; // 本核处理的元素数
    int64_t tileLength = 0;  // 单块元素数
};

template <typename DT_X>
__aicore__ inline void Relu<DT_X>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t offset = blockIdx * tilingData->blockFactor;
    // 防御：本核起始位置已越过总元素数（多核切分的尾部核）直接空转
    if (offset >= tilingData->totalNum) {
        blockLength = 0;
        tileLength = 0;
        return;
    }
    // 本核实际处理长度：最后一个核可能不足 blockFactor，按总元素数截断
    int64_t len = LessOf(tilingData->blockFactor, tilingData->totalNum - offset);
    blockLength = len;
    tileLength = (tilingData->ubFactor > 0 && tilingData->ubFactor <= len) ? tilingData->ubFactor : len;

    // 设置每个核的 Global Memory 起始地址，第三个参数为本核的有效元素数，用于越界保护
    inputGM.SetGlobalBuffer((__gm__ DT_X*)x + offset, len);
    outputGM.SetGlobalBuffer((__gm__ DT_X*)y + offset, len);
    // 为队列分配 UB 内存
    pipe.InitBuffer(inputQueue, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, this->tileLength * sizeof(DT_X));
}

template <typename DT_X>
__aicore__ inline void Relu<DT_X>::Process()
{
    if (blockLength <= 0 || tileLength <= 0) {
        return;
    }
    int64_t loopCount = (blockLength + tileLength - 1) / tileLength;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t progress = i * tileLength;
        // 尾块长度必须逐块计算，并一路传给 Compute/CopyOut，
        // 否则最后一个块会按整块长度计算与写回，读到未初始化 UB 并越界写 GM
        int64_t currentNum = LessOf(tileLength, blockLength - progress);
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

template <typename DT_X>
__aicore__ inline void Relu<DT_X>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<DT_X> xLocal = inputQueue.AllocTensor<DT_X>();
    if (CanUsePlainCopy<DT_X>(currentNum)) {
        DataCopy(xLocal, inputGM[progress], static_cast<uint32_t>(currentNum));
    } else {
        DataCopyParams copySpec;
        copySpec.blockCount = 1;
        copySpec.blockLen = static_cast<uint32_t>(currentNum * sizeof(DT_X));
        copySpec.srcStride = 0;
        copySpec.dstStride = 0;
        DataCopyPad(xLocal, inputGM[progress], copySpec, {false, 0, 0, 0});
    }
    inputQueue.EnQue(xLocal);
}

template <typename DT_X>
__aicore__ inline void Relu<DT_X>::Compute(int64_t currentNum)
{
    LocalTensor<DT_X> xLocal = inputQueue.DeQue<DT_X>();
    LocalTensor<DT_X> yLocal = outputQueue.AllocTensor<DT_X>();
    AscendC::Relu(yLocal, xLocal, static_cast<uint32_t>(currentNum));
    outputQueue.EnQue<DT_X>(yLocal);
    inputQueue.FreeTensor(xLocal);
}

template <typename DT_X>
__aicore__ inline void Relu<DT_X>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<DT_X> yLocal = outputQueue.DeQue<DT_X>();
    if (CanUsePlainCopy<DT_X>(currentNum)) {
        DataCopy(outputGM[progress], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyParams copySpec;
        copySpec.blockCount = 1;
        copySpec.blockLen = static_cast<uint32_t>(currentNum * sizeof(DT_X));
        copySpec.srcStride = 0;
        copySpec.dstStride = 0;
        DataCopyPad(outputGM[progress], yLocal, copySpec);
    }
    outputQueue.FreeTensor(yLocal);
}

} // namespace NsRelu
#endif // RELU_H
