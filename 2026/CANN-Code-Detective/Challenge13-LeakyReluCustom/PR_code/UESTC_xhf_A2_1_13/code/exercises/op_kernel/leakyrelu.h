/*!
 * \file leakyrelu.h
 * \brief Leakyrelu 算子 kernel 类定义
 */

#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "leakyrelu_tiling_data.h"
#include "leakyrelu_tiling_key.h"

namespace NsLeakyrelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Leakyrelu {
public:
    __aicore__ inline Leakyrelu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData);
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
    float negativeSlope_ = 0.01F;
};

template <typename T>
__aicore__ inline void Leakyrelu<T>::Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
{
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    negativeSlope_ = tilingData->negativeSlope;
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    auto x = inputQueueX.AllocTensor<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(x, inputGMX[progress * ubLength_], params, pad);
    inputQueueX.EnQue(x);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Compute(int64_t currentNum)
{
    auto x = inputQueueX.DeQue<T>();
    auto y = outputQueueY.AllocTensor<T>();
    AscendC::LeakyRelu(y, x, static_cast<T>(negativeSlope_), static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(y);
    inputQueueX.FreeTensor(x);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    auto y = outputQueueY.DeQue<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], y, params);
    outputQueueY.FreeTensor(y);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    for (int64_t i = 0; i * ubLength_ < blockLength_; ++i) {
        const int64_t remaining = blockLength_ - i * ubLength_;
        const int64_t count = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(i, count);
        Compute(count);
        CopyOut(i, count);
    }
}

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
