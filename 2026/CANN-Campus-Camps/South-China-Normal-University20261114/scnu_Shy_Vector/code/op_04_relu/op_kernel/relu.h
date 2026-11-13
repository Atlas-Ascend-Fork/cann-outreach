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

constexpr int32_t IN_BUFFER_NUM = 2;  // 输入队列 buffer 份数（DoubleBuffer）
constexpr int32_t OUT_BUFFER_NUM = 2; // 输出队列 buffer 份数（DoubleBuffer）
constexpr int64_t BLOCK_BYTE_SIZE = 32; // 256bit，对齐搬运的块粒度
constexpr int64_t UB_PAD_SIZE = 32;     // 尾块非对齐补齐，避免 UB 越界访问

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyInAll(int64_t progress);
    __aicore__ inline void CopyOutAll(int64_t progress);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, IN_BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, OUT_BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t alignElements_ = 0;
    int64_t startIdx_ = 0;    // 本核负责处理的元素起点（Global Memory 起始地址）
    int64_t coreDataNum_ = 0; // 本核负责处理的元素个数
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    alignElements_ = BLOCK_BYTE_SIZE / static_cast<int64_t>(sizeof(T));

    // 按核号分配 GM 起始地址：第 blockIdx 个核从 blockIdx * blockFactor 处开始处理
    startIdx_ = static_cast<int64_t>(GetBlockIdx()) * blockLength_;
    const int64_t remain = tilingData->totalNum - startIdx_;
    coreDataNum_ = remain > 0 ? (remain < blockLength_ ? remain : blockLength_) : 0;

    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T) + UB_PAD_SIZE);
    pipe.InitBuffer(inputQueueX, IN_BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, OUT_BUFFER_NUM, bufferBytes);

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xTensor = inputQueueX.AllocTensor<T>();
    const int64_t alignNum = currentNum / alignElements_ * alignElements_;
    if (alignNum > 0) { // 对齐部分按 32B 块粒度搬运
        DataCopy(xTensor, inputGMX[progress], static_cast<uint32_t>(alignNum));
    }
    if (currentNum > alignNum) { // 尾块按非对齐通路搬运，读取长度精确到元素个数
        DataCopyExtParams copyParams{1, static_cast<uint32_t>((currentNum - alignNum) * sizeof(T)), 0, 0, 0};
        DataCopyPad(xTensor[alignNum], inputGMX[progress + alignNum], copyParams, DataCopyPadExtParams<T>{});
    }
    inputQueueX.EnQue(xTensor);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyInAll(int64_t progress)
{
    // 整块快速通路：ubLength_ 由 tiling 保证按 32B 对齐，且 GM 偏移同样对齐，可直接整块搬运
    LocalTensor<T> xTensor = inputQueueX.AllocTensor<T>();
    DataCopy(xTensor, inputGMX[progress], static_cast<uint32_t>(ubLength_));
    inputQueueX.EnQue(xTensor);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOutAll(int64_t progress)
{
    LocalTensor<T> yTensor = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], yTensor, static_cast<uint32_t>(ubLength_));
    outputQueueY.FreeTensor(yTensor);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xTensor = inputQueueX.DeQue<T>();
    LocalTensor<T> yTensor = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yTensor, xTensor, static_cast<int32_t>(currentNum)); // dst = max(src, 0)
    inputQueueX.FreeTensor(xTensor);
    outputQueueY.EnQue(yTensor);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yTensor = outputQueueY.DeQue<T>();
    const int64_t alignNum = currentNum / alignElements_ * alignElements_;
    if (alignNum > 0) { // 对齐部分按 32B 块粒度搬运
        DataCopy(outputGMY[progress], yTensor, static_cast<uint32_t>(alignNum));
    }
    if (currentNum > alignNum) { // 尾块按非对齐通路搬运，写出长度精确到元素个数
        DataCopyExtParams copyParams{1, static_cast<uint32_t>((currentNum - alignNum) * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[progress + alignNum], yTensor[alignNum], copyParams);
    }
    outputQueueY.FreeTensor(yTensor);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (coreDataNum_ <= 0) {
        return;
    }
    const int64_t fullLoops = coreDataNum_ / ubLength_; // 整块循环次数（数据量按 32B 对齐切分）
    for (int64_t i = 0; i < fullLoops; ++i) {
        const int64_t progress = startIdx_ + i * ubLength_;
        CopyInAll(progress);
        Compute(ubLength_);
        CopyOutAll(progress);
    }
    const int64_t tailNum = coreDataNum_ - fullLoops * ubLength_;
    if (tailNum > 0) { // 尾块按精确元素个数走非对齐通路
        const int64_t progress = startIdx_ + fullLoops * ubLength_;
        CopyIn(progress, tailNum);
        Compute(tailNum);
        CopyOut(progress, tailNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
