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

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    int64_t blockFactor = tilingData->blockFactor;
    int64_t ubFactor    = tilingData->ubFactor;
    int64_t totalNum    = tilingData->totalNum;

    int64_t blockIdx  = static_cast<int64_t>(GetBlockIdx());
    int64_t remainder = totalNum - blockIdx * blockFactor;
    
    // 计算当前核实际需要处理的数据长度
    blockLength_ = (remainder >= blockFactor) ? blockFactor
                 : (remainder > 0 ? remainder : 0);

    if (blockLength_ <= 0) {
        blockLength_ = 0;
        return;
    }

    // 若当前核剩余数据量小于 ubFactor，则调整 ubLength_ 以节省 UB 空间
    ubLength_ = (ubFactor > blockLength_) ? blockLength_ : ubFactor;

    // 设置每个核的 Global Memory 起始地址
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockIdx * blockFactor, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockIdx * blockFactor, blockLength_);

    // 初始化 UB Buffer，按 32 字节对齐分配
    int64_t ubBytes = ((ubLength_ * (int64_t)sizeof(T) + 31) / 32) * 32;
    pipe.InitBuffer(inputQueueX,  BUFFER_NUM, ubBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    int64_t offset = progress * ubLength_;
    constexpr int32_t alignElements = 32 / sizeof(T);

    if (currentNum % alignElements == 0) {
        // 32B 对齐，直接搬运
        DataCopy(xLocal, inputGMX[offset], currentNum);
    } else {
        // 非对齐，使用 DataCopyPad
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen   = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride  = 0;
        copyParams.dstStride  = 0;
        DataCopyPadExtParams<T> padParams;
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // 实现 Relu 计算逻辑：y = max(0, x)
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    int64_t offset = progress * ubLength_;
    constexpr int32_t alignElements = 32 / sizeof(T);

    if (currentNum % alignElements == 0) {
        DataCopy(outputGMY[offset], yLocal, currentNum);
    } else {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen   = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride  = 0;
        copyParams.dstStride  = 0;
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    // 循环处理当前核负责的所有数据
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (i == loopCount - 1)
                           ? (blockLength_ - i * ubLength_)
                           : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H