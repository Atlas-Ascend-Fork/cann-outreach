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

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    TBuf<TPosition::VECCALC> posBuf, negBuf, sumBuf, maskBuf;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(posBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(negBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(sumBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(maskBuf, ubLength_ / 8);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    auto input = inputQueueX.AllocTensor<T>();
    DataCopy(input, inputGMX[progress], currentNum);
    inputQueueX.EnQue(input);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    auto input = inputQueueX.DeQue<T>();
    auto output = outputQueueY.AllocTensor<T>();
    auto pos = posBuf.Get<float>();
    auto neg = negBuf.Get<float>();
    auto sum = sumBuf.Get<float>();
    const uint32_t count = static_cast<uint32_t>(currentNum);
    if constexpr (sizeof(T) == sizeof(float)) {
        Muls(pos, input, 1.0f, count);
    } else {
        Cast(pos, input, RoundMode::CAST_NONE, count);
    }
    Mins(pos, pos, 10.0f, count);
    Maxs(pos, pos, -10.0f, count);
    Muls(neg, pos, -1.0f, count);
    Exp(pos, pos, count);
    Exp(neg, neg, count);
    Add(sum, pos, neg, count);
    Sub(pos, pos, neg, count);
    Div(pos, pos, sum, count);
    if constexpr (sizeof(T) == sizeof(float)) {
        Muls(neg, input, 1.0f, count);
    } else {
        Cast(neg, input, RoundMode::CAST_NONE, count);
    }
    Abs(sum, neg, count);
    auto mask = maskBuf.Get<uint8_t>();
    CompareScalar(mask, sum, 0.001f, CMPMODE::LE, count);
    Select(pos, mask, neg, pos, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    if constexpr (sizeof(T) == sizeof(float)) {
        Muls(output, pos, 1.0f, count);
    } else {
        Cast(output, pos, RoundMode::CAST_RINT, count);
    }
    outputQueueY.EnQue(output);
    inputQueueX.FreeTensor(input);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    auto output = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], output, currentNum);
    outputQueueY.FreeTensor(output);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
        const int64_t remaining = blockLength_ - offset;
        const int64_t count = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(offset, count);
        Compute(count);
        CopyOut(offset, count);
    }
}

} // namespace NsTanh
#endif // TANH_H
