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

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    const int64_t start = GetBlockIdx() * blockLength_;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    auto src = inputQueueX.AllocTensor<T>();
    DataCopy(src, inputGMX[progress], currentNum);
    inputQueueX.EnQue(src);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    auto src = inputQueueX.DeQue<T>();
    auto dst = outputQueueY.AllocTensor<T>();
    AscendC::Relu(dst, src, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue(dst);
    inputQueueX.FreeTensor(src);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    auto dst = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], dst, currentNum);
    outputQueueY.FreeTensor(dst);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    for (int64_t i = 0; i < blockLength_; i += ubLength_) {
        const int64_t count = blockLength_ - i < ubLength_ ? blockLength_ - i : ubLength_;
        CopyIn(i, count);
        Compute(count);
        CopyOut(i, count);
    }
}

} // namespace NsRelu
#endif // RELU_H
