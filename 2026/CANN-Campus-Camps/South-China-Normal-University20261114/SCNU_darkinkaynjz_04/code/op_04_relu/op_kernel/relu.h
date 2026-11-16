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
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    auto input = inputQueueX.AllocTensor<T>();
    DataCopy(input, inputGMX[progress], currentNum);
    inputQueueX.EnQue(input);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    auto input = inputQueueX.DeQue<T>();
    auto output = outputQueueY.AllocTensor<T>();
    Maxs(output, input, static_cast<T>(0), currentNum);
    outputQueueY.EnQue(output);
    inputQueueX.FreeTensor(input);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    auto output = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], output, currentNum);
    outputQueueY.FreeTensor(output);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
        const int64_t remaining = blockLength_ - offset;
        const int64_t count = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(offset, count);
        Compute(count);
        CopyOut(offset, count);
    }
}

} // namespace NsRelu
#endif // RELU_H
