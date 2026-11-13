/*!
 * \file leakyrelu.h
 * \brief Leakyrelu 算子 kernel 类定义
 *
 * y = max(x, 0) + negative_slope * min(x, 0)
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
    T negativeSlope_ = static_cast<T>(0);
};

template <typename T>
__aicore__ inline void Leakyrelu<T>::Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
{
    ubLength_ = tilingData->ubFactor;
    negativeSlope_ = static_cast<T>(tilingData->negativeSlope);

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t offset = blockIdx * tilingData->blockFactor;
    int64_t length = tilingData->totalNum - offset;
    if (length < 0) {
        length = 0;
    }
    if (length > tilingData->blockFactor) {
        length = tilingData->blockFactor;
    }
    blockLength_ = length;

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);

    if (ubLength_ > 0) {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    }
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // y = max(x, 0) + negative_slope * min(x, 0)
    Mins(yLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));
    Muls(yLocal, yLocal, negativeSlope_, static_cast<int32_t>(currentNum));
    Maxs(xLocal, xLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));
    Add(yLocal, yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    for (int64_t i = 0; i < blockLength_; i += ubLength_) {
        int64_t currentNum = blockLength_ - i;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
