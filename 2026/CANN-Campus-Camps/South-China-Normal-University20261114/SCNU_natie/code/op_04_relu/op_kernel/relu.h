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
// 32B 一个 block，作为 GM <-> UB 搬运的对齐粒度
constexpr int64_t BLOCK_BYTES = 32;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);
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
    // 每个核负责的 GM 区间：从 blockIdx * blockFactor 开始，长度 blockFactor（尾核可能不足）
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t startOffset = blockIdx * blockFactor;

    blockLength_ = tilingData->totalNum - startOffset;
    if (blockLength_ > blockFactor) {
        blockLength_ = blockFactor;
    }
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(x) + startOffset, static_cast<uint64_t>(blockLength_));
    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(y) + startOffset, static_cast<uint64_t>(blockLength_));

    if (ubLength_ > 0) {
        const uint32_t ubBytes = static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T)));
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubBytes);
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubBytes);
    }
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    constexpr int64_t alignNum = BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    if (currentNum % alignNum == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<int32_t>(currentNum));
    } else {
        // 尾块不足 32B 时用 DataCopyPad，避免非对齐搬运
        DataCopyExtParams copyParams{};
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPadExtParams<T> padParams{};
        padParams.isPad = false;

        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // y = max(x, 0)，向量指令 vmaxs
    AscendC::Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    constexpr int64_t alignNum = BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    if (currentNum % alignNum == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<int32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{};
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (ubLength_ <= 0) {
        return;
    }
    // 循环内 CopyIn / Compute / CopyOut 配合深度为 2 的 TQue 形成 DoubleBuffer，
    // 相邻 tile 的搬运与计算可以流水并行
    int64_t offset = 0;
    while (offset < blockLength_) {
        int64_t currentNum = blockLength_ - offset;
        currentNum = (currentNum > ubLength_) ? ubLength_ : currentNum;
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
        offset += currentNum;
    }
}

} // namespace NsRelu
#endif // RELU_H