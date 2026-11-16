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
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(
        int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    // 计算全部在 float32 中进行。
    TBuf<TPosition::VECCALC> workBuffer;
    TBuf<TPosition::VECCALC> expNegBuffer;
    TBuf<TPosition::VECCALC> numeratorBuffer;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(
    GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t offset =
        static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;

    ubLength_ = tilingData->ubFactor;

    if (offset >= tilingData->totalNum) {
        blockLength_ = 0;
        return;
    }

    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor
        ? remaining
        : tilingData->blockFactor;

    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);

    pipe.InitBuffer(
        inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(
        outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));

    pipe.InitBuffer(workBuffer, ubLength_ * sizeof(float));
    pipe.InitBuffer(expNegBuffer, ubLength_ * sizeof(float));
    pipe.InitBuffer(numeratorBuffer, ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;

    // 非 32 字节对齐的尾块使用 DataCopyPad。
    if ((currentNum * sizeof(T)) % 32 == 0) {
        DataCopy(xLocal, inputGMX[offset],
                 static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(currentNum * sizeof(T)),
            0,
            0,
            0
        };
        DataCopyPadExtParams<T> padParams{
            false, 0, 0, static_cast<T>(0)
        };
        DataCopyPad(
            xLocal, inputGMX[offset], copyParams, padParams);
    }

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    LocalTensor<float> work = workBuffer.Get<float>();
    LocalTensor<float> expNeg = expNegBuffer.Get<float>();
    LocalTensor<float> numerator = numeratorBuffer.Get<float>();

    const uint32_t count = static_cast<uint32_t>(currentNum);

    if constexpr (sizeof(T) == sizeof(float)) {
        Adds(work, xLocal, 0.0f, count);
    } else {
        Cast(work, xLocal, RoundMode::CAST_NONE, count);
    }

    // tanh 在大幅值处趋近 ±1。
    // 限制指数输入，避免直接计算 exp(x) 时发生溢出。
    Maxs(work, work, -10.0f, count);
    Mins(work, work, 10.0f, count);

    // expNeg = exp(-x)
    Muls(expNeg, work, -1.0f, count);
    Exp(expNeg, expNeg, count);

    // work = exp(x)
    Exp(work, work, count);

    // numerator = exp(x) - exp(-x)
    Sub(numerator, work, expNeg, count);

    // work = exp(x) + exp(-x)
    Add(work, work, expNeg, count);

    // numerator = tanh(x)
    Div(numerator, numerator, work, count);

    if constexpr (sizeof(T) == sizeof(float)) {
        Adds(yLocal, numerator, 0.0f, count);
    } else {
        Cast(yLocal, numerator, RoundMode::CAST_RINT, count);
    }

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;

    if ((currentNum * sizeof(T)) % 32 == 0) {
        DataCopy(outputGMY[offset], yLocal,
                 static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(currentNum * sizeof(T)),
            0,
            0,
            0
        };
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    const int64_t loopCount =
        (blockLength_ + ubLength_ - 1) / ubLength_;

    for (int64_t progress = 0; progress < loopCount; ++progress) {
        const int64_t remaining =
            blockLength_ - progress * ubLength_;
        const int64_t currentNum =
            remaining < ubLength_ ? remaining : ubLength_;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsTanh

#endif // TANH_H
