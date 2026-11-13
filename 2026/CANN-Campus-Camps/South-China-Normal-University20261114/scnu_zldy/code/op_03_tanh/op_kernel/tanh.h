/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> workBuffer;
    TBuf<QuePosition::VECCALC> numeratorBuffer;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    ubLength_ = tilingData->ubFactor;
    const int64_t start = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - start;
    blockLength_ = remaining > 0 ? remaining : 0;
    if (blockLength_ > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(workBuffer, ubLength_ * sizeof(float));
    pipe.InitBuffer(numeratorBuffer, ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], params, padding);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> work = workBuffer.Get<float>();
    LocalTensor<float> numerator = numeratorBuffer.Get<float>();
    const uint32_t count = static_cast<uint32_t>(currentNum);
    if constexpr (sizeof(T) == sizeof(float)) {
        Adds(work, xLocal, 0.0f, count);
    } else {
        Cast(work, xLocal, RoundMode::CAST_NONE, count);
    }
    // tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1). Float32 intermediates
    // avoid half-precision overflow. Beyond +/-10 the float result
    // already rounds to +/-1, so clipping also bounds Exp's argument.
    Maxs(work, work, -10.0f, count);
    Mins(work, work, 10.0f, count);
    Muls(work, work, 2.0f, count);
    Exp(work, work, count);
    Adds(numerator, work, -1.0f, count);
    Adds(work, work, 1.0f, count);
    Div(work, numerator, work, count);
    if constexpr (sizeof(T) == sizeof(float)) {
        Adds(yLocal, work, 0.0f, count);
    } else {
        Cast(yLocal, work, RoundMode::CAST_RINT, count);
    }
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, params);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    for (int64_t progress = 0; progress * ubLength_ < blockLength_; ++progress) {
        const int64_t remaining = blockLength_ - progress * ubLength_;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
