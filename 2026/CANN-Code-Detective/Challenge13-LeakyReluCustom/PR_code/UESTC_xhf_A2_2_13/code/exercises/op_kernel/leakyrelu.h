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
    __aicore__ inline Leakyrelu() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> negativeBuffer;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    T negativeSlope_ = static_cast<T>(0.01f);
};

template <typename T>
__aicore__ inline void Leakyrelu<T>::Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
{
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining > 0 ?
        (remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor) : 0;
    ubLength_ = tilingData->ubFactor;
    negativeSlope_ = static_cast<T>(tilingData->negativeSlope);
    if (blockLength_ == 0 || ubLength_ <= 0) {
        return;
    }
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(negativeBuffer, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    if ((currentNum * sizeof(T)) % 32 == 0) {
        DataCopy(xLocal, inputGMX[progress], currentNum);
    } else {
        DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        DataCopyPad(xLocal, inputGMX[progress], params, padding);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    const auto count = static_cast<int32_t>(currentNum);
    const T zero = static_cast<T>(0);
    LocalTensor<T> negative = negativeBuffer.Get<T>();
    Maxs(yLocal, xLocal, zero, count);
    Mins(negative, xLocal, zero, count);
    Muls(negative, negative, negativeSlope_, count);
    Add(yLocal, yLocal, negative, count);
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    if ((currentNum * sizeof(T)) % 32 == 0) {
        DataCopy(outputGMY[progress], yLocal, currentNum);
    } else {
        DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[progress], yLocal, params);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Process()
{
    if (ubLength_ <= 0) {
        return;
    }
    for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
        const int64_t remaining = blockLength_ - offset;
        const int64_t count = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(offset, count);
        Compute(count);
        CopyOut(offset, count);
    }
}

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
