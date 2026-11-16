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

    int64_t blockLength_ = 0;   // 每核处理的基础元素数
    int64_t ubLength_ = 0;      // 单次搬运/计算的元素数
    int64_t totalLength_ = 0;   // 总元素数，用于尾核/尾块判断
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 直接从 tilingData 读取，字段名与新 ReluTilingData 保持一致
    totalLength_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    int64_t gmOffset = static_cast<int64_t>(blockIdx) * blockLength_;

    // 多核切分：每个核的 GM 起始地址按 blockLength_ 偏移
    inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, blockLength_);

    // 为队列分配 UB 内存（双缓冲）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {});

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // ReLU：y = max(x, 0)
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    uint32_t blockIdx = AscendC::GetBlockIdx();

    // 本核实际需要处理的长度（最后一个核可能不满）
    int64_t start = static_cast<int64_t>(blockIdx) * blockLength_;
    int64_t remain = totalLength_ - start;
    if (remain <= 0) {
        return;
    }
    int64_t currentBlockLength = (remain < blockLength_) ? remain : blockLength_;

    // 循环次数 = ceil(currentBlockLength / ubLength_)
    int64_t loopCount = (currentBlockLength + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = ubLength_;
        if (i == loopCount - 1) {
            currentNum = currentBlockLength - i * ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H