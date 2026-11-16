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

    int64_t blockOffset_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// 初始化
template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // 获取当前核编号和负责的数据范围
    int64_t blockIdx = GetBlockIdx();
    blockOffset_ = blockIdx * tilingData->blockFactor;

    int64_t remainNum = tilingData->totalNum - blockOffset_;
    blockLength_ = (remainNum > tilingData->blockFactor)
        ? tilingData->blockFactor : remainNum;

    ubLength_ = tilingData->ubFactor;

    // 设置Global Memory起始地址
    inputGMX.SetGlobalBuffer((__gm__ T*)x, tilingData->totalNum);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, tilingData->totalNum);

    // 初始化输入输出队列
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

// 数据搬入
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(currentNum * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPad(
        xLocal,
        inputGMX[blockOffset_ + progress],
        copyParams,
        padParams);

    inputQueueX.EnQue(xLocal);
}

// ReLU计算
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    //Relu(yLocal, xLocal, currentNum);
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));

    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue(yLocal);
}

// 数据搬出
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(currentNum * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPad(
        outputGMY[blockOffset_ + progress],
        yLocal,
        copyParams
        );//删除了第四个参数padParams

    outputQueueY.FreeTensor(yLocal);
}

// 计算流程
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    for (int64_t progress = 0; progress < blockLength_; progress += ubLength_) {
        int64_t currentNum = (blockLength_ - progress > ubLength_)
            ? ubLength_ : blockLength_ - progress;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu

#endif // RELU_H
