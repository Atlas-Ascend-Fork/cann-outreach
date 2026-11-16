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
constexpr int64_t UB_ALIGN_BYTES = 32;

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

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t totalNum = tilingData->totalNum;

    // 本核负责的 GM 区间：起点 = blockIdx * blockFactor，长度取 blockFactor
    int64_t offset = blockIdx * blockFactor;
    if (offset > totalNum) {
        offset = totalNum;
    }
    int64_t remain = totalNum - offset;
    blockLength_ = (remain < blockFactor) ? remain : blockFactor;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        // 防御：tile 长度至少给一个 UB 块，避免 InitBuffer 长度为 0
        ubLength_ = UB_ALIGN_BYTES / static_cast<int64_t>(sizeof(T));
    }

    // 每个核绑定属于自己的 GM 起始地址，核内所有 tile 都基于该基址偏移
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);

    // 每个队列 BUFFER_NUM 份 buffer -> DoubleBuffer
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t gmOffset = progress * ubLength_;
    const int64_t alignNum = UB_ALIGN_BYTES / static_cast<int64_t>(sizeof(T));

    if (currentNum % alignNum == 0) {
        // 整块 32B 对齐：走无 Pad 的快路径
        DataCopy(xLocal, inputGMX[gmOffset], static_cast<uint32_t>(currentNum));
    } else {
        // 尾部 tile：对齐部分正常搬运，剩余不足 32B 的部分用 DataCopyPad 补齐
        const int64_t alignedNum = currentNum / alignNum * alignNum;
        if (alignedNum > 0) {
            DataCopy(xLocal, inputGMX[gmOffset], static_cast<uint32_t>(alignedNum));
        }
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>((currentNum - alignedNum) * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = static_cast<T>(0);
        DataCopyPad(xLocal[alignedNum], inputGMX[gmOffset + alignedNum], copyParams, padParams);
    }

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // y = max(0, x)，一条向量指令完成（fp16 / fp32 均支持）
    Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t gmOffset = progress * ubLength_;
    const int64_t alignNum = UB_ALIGN_BYTES / static_cast<int64_t>(sizeof(T));

    if (currentNum % alignNum == 0) {
        DataCopy(outputGMY[gmOffset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        const int64_t alignedNum = currentNum / alignNum * alignNum;
        if (alignedNum > 0) {
            DataCopy(outputGMY[gmOffset], yLocal, static_cast<uint32_t>(alignedNum));
        }
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>((currentNum - alignedNum) * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(outputGMY[gmOffset + alignedNum], yLocal[alignedNum], copyParams);
    }

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // TODO: 实现 Process 逻辑
    //核内按 ubLength_ 切成若干 tile，循环体是标准的 DoubleBuffer 结构,队列深度为 2，第 i+1 次 AllocTensor 不会被阻塞，从而让第 i 次的Compute/CopyOut 与第 i+1 次的 CopyIn 在 MTE2 / V / MTE3 上并行
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
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
