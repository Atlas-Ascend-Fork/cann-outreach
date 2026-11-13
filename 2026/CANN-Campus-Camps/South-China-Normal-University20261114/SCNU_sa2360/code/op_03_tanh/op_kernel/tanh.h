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
    TBuf<TPosition::VECCALC> expXBuf;
    TBuf<TPosition::VECCALC> expNegXBuf;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * blockLength_, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + GetBlockIdx() * blockLength_, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(expXBuf, ubLength_ * sizeof(T));
    pipe.InitBuffer(expNegXBuf, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<T> expX = expXBuf.Get<T>();
    LocalTensor<T> expNegX = expNegXBuf.Get<T>();

    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Exp(expX, xLocal, currentNum);
    Muls(expNegX, xLocal, static_cast<T>(-1.0f), currentNum);
    Exp(expNegX, expNegX, currentNum);
    Sub(yLocal, expX, expNegX, currentNum);
    Add(expX, expX, expNegX, currentNum);
    Div(yLocal, yLocal, expX, currentNum);

    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue<T>(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    for (int64_t progress = 0; progress < blockLength_; progress += ubLength_) {
        int64_t currentNum = blockLength_ - progress;
        if (currentNum > ubLength_) currentNum = ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}
} // namespace NsTanh
#endif // TANH_H
