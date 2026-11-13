
/*!
 * \file tanh.h
 * \brief Tanh operator kernel (composed from basic Ascend C vector APIs).
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
    __aicore__ inline Tanh() {}

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

    // Float32 temporary buffers used during computation.
    TBuf<QuePosition::VECCALC> inputFloatBuf;
    TBuf<QuePosition::VECCALC> expBuf;
    TBuf<QuePosition::VECCALC> numeratorBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const TanhTilingData* tilingData)
{
    const int64_t start =
        static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;

    const int64_t remaining = tilingData->totalNum - start;

    if (remaining <= 0) {
        return;
    }

    blockLength_ = (remaining < tilingData->blockFactor)
                       ? remaining
                       : tilingData->blockFactor;

    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, blockLength_);

    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ * sizeof(T));

    pipe.InitBuffer(
        inputFloatBuf,
        ubLength_ * sizeof(float));

    pipe.InitBuffer(
        expBuf,
        ubLength_ * sizeof(float));

    pipe.InitBuffer(
        numeratorBuf,
        ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    DataCopy(
        xLocal,
        inputGMX[progress],
        static_cast<uint32_t>(currentNum));

    inputQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    LocalTensor<float> expLocal = expBuf.Get<float>();
    LocalTensor<float> numeratorLocal = numeratorBuf.Get<float>();

    LocalTensor<float> xFloat =
        xLocal.template ReinterpretCast<float>();

    // Convert float16 input to float32 for computation.
    if constexpr (sizeof(T) == sizeof(half)) {
        xFloat = inputFloatBuf.Get<float>();

        Cast(
            xFloat,
            xLocal,
            RoundMode::CAST_NONE,
            static_cast<uint32_t>(currentNum));
    }

    // tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
    //
    // Clamp input to [-40, 40] to avoid exponential overflow.

    Mins(
        expLocal,
        xFloat,
        40.0f,
        static_cast<uint32_t>(currentNum));

    Maxs(
        expLocal,
        expLocal,
        -40.0f,
        static_cast<uint32_t>(currentNum));

    Muls(
        expLocal,
        expLocal,
        2.0f,
        static_cast<uint32_t>(currentNum));

    Exp(
        expLocal,
        expLocal,
        static_cast<uint32_t>(currentNum));

    // numerator = exp(2*x) - 1
    Adds(
        numeratorLocal,
        expLocal,
        -1.0f,
        static_cast<uint32_t>(currentNum));

    // denominator = exp(2*x) + 1
    Adds(
        expLocal,
        expLocal,
        1.0f,
        static_cast<uint32_t>(currentNum));

    if constexpr (sizeof(T) == sizeof(half)) {

        // Compute result in float32.
        Div(
            xFloat,
            numeratorLocal,
            expLocal,
            static_cast<uint32_t>(currentNum));

        // Convert result back to float16.
        Cast(
            yLocal,
            xFloat,
            RoundMode::CAST_RINT,
            static_cast<uint32_t>(currentNum));

    } else {

        LocalTensor<float> yFloat =
            yLocal.template ReinterpretCast<float>();

        Div(
            yFloat,
            numeratorLocal,
            expLocal,
            static_cast<uint32_t>(currentNum));
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    DataCopy(
        outputGMY[progress],
        yLocal,
        static_cast<uint32_t>(currentNum));

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ == 0 || ubLength_ == 0) {
        return;
    }

    for (int64_t progress = 0;
         progress < blockLength_;
         progress += ubLength_) {

        const int64_t remaining = blockLength_ - progress;

        const int64_t currentNum =
            (remaining < ubLength_) ? remaining : ubLength_;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsTanh

#endif // TANH_H
