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
    int64_t blockIdx = GetBlockIdx();
    int64_t offset = blockIdx * tilingData->blockFactor;  // 本核 GM 元素偏移（int64 无溢出）
    // 尾核实际长度可能 < blockFactor；totalNum=0 或空块时 blockLength_=0（Process 空转返回）
    blockLength_ = (tilingData->totalNum - offset) < tilingData->blockFactor
                       ? (tilingData->totalNum - offset) : tilingData->blockFactor;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    Maxs(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));
    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;  // 空块防护（totalNum=0 或空 shape）
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;

    // 预取首块（双缓冲流水：MTE2 下一块搬运 与 Vector 当前块计算 重叠）
    int64_t firstNum = blockLength_ < ubLength_ ? blockLength_ : ubLength_;
    CopyIn(0, firstNum);

    for (int64_t progress = 0; progress < loopCount; progress++) {
        int64_t blockRemain = blockLength_ - progress * ubLength_;
        int64_t currentNum = blockRemain < ubLength_ ? blockRemain : ubLength_;
        if (progress + 1 < loopCount) {
            int64_t nextRemain = blockLength_ - (progress + 1) * ubLength_;
            int64_t nextNum = nextRemain < ubLength_ ? nextRemain : ubLength_;
            CopyIn(progress + 1, nextNum);  // 预取下一块
        }
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H
