/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 *
 * 实现要点：
 *   1. 每个核根据 blockIdx 计算自己在 Global Memory 上的起始地址与数据长度，
 *      多核并行处理整块输入；
 *   2. 核内按 ubFactor 分 tile，采用 TQue + DoubleBuffer(BUFFER_NUM = 2)
 *      让 MTE2(搬入)、Vector(计算)、MTE3(搬出) 流水并行；
 *   3. 尾块用 DataCopyPad 搬运，兼容任意非对齐的输入元素总数。
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

// DoubleBuffer 深度：同一个队列上挂 2 块 buffer，搬入/计算/搬出可流水并行
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

    int64_t blockLength_ = 0;  // 当前核实际处理的元素个数
    int64_t ubLength_ = 0;     // 单个 tile 的元素个数
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    const int64_t totalNum = tilingData->totalNum;
    const int64_t blockFactor = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    // 当前核在 Global Memory 上的起始偏移：blockIdx * blockFactor
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t offset = blockIdx * blockFactor;

    // 最后一个核可能不足 blockFactor，需要按剩余元素数截断
    blockLength_ = totalNum - offset;
    if (blockLength_ > blockFactor) {
        blockLength_ = blockFactor;
    }
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    // 没有数据的核直接返回，不申请 UB
    if (blockLength_ == 0) {
        return;
    }
    // 防御性处理：ubFactor 异常时至少保证一个元素
    if (ubLength_ < 1) {
        ubLength_ = 1;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    // DoubleBuffer：输入/输出队列各 BUFFER_NUM 块，每块 ubLength_ 个元素；
    // 字节数按 32B 向上对齐，保证 DMA 访问不越过 buffer 尾部
    const int64_t bufferBytes = ubLength_ * static_cast<int64_t>(sizeof(T));
    const uint32_t bufferSize = static_cast<uint32_t>((bufferBytes + 31) / 32 * 32);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferSize);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferSize);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    // 单 block 连续搬运 currentNum 个元素，非 32B 对齐的尾块由 DataCopyPad 补齐
    const uint32_t copyBytes = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T)));
    if (copyBytes % 32U == 0U) {
        // 32B 对齐时走普通 DataCopy 快路径，省去 DataCopyPad 的补齐处理
        DataCopy(inputLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGMX[progress], copyParams, padParams);
    }
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();

    // y = max(0, x)，使用 AscendC 单目指令 ReLU 接口；
    // 由于本文件中的类模板同样名为 Relu，此处必须用 AscendC:: 限定函数查找
    AscendC::Relu(outputLocal, inputLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    const uint32_t copyBytes = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T)));
    if (copyBytes % 32U == 0U) {
        // 32B 对齐时走普通 DataCopy 快路径
        DataCopy(outputGMY[progress], outputLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
        DataCopyPad(outputGMY[progress], outputLocal, copyParams);
    }
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    // 当前核按 ubLength_ 切成 tile 循环处理，最后一个 tile 处理余数
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        const int64_t offset = i * ubLength_;
        const int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - offset) : ubLength_;
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
