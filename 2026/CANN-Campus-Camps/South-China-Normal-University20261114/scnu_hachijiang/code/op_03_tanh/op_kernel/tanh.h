/*!
 * \file tanh.h
 * \brief Tanh kernel class definition
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
    // temp buffers for the tanh computation
    TBuf<TPosition::VECCALC> tempBufExpX;
    TBuf<TPosition::VECCALC> tempBufExpNegX;
    TBuf<TPosition::VECCALC> tempBufNum;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;  // elements handled by this core
    int64_t ubLength_ = 0;     // elements handled by one UB iteration
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(tempBufExpX, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(tempBufExpNegX, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(tempBufNum, static_cast<uint32_t>(ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// x is clamped to [-8, 8] first, so exp() never overflows (tanh saturates there)
template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    int64_t alignElems = 32 / sizeof(T);
    int64_t calcCount = ((currentNum + alignElems - 1) / alignElems) * alignElems;

    AscendC::LocalTensor<T> expX = tempBufExpX.Get<T>();
    AscendC::LocalTensor<T> expNegX = tempBufExpNegX.Get<T>();
    AscendC::LocalTensor<T> numTensor = tempBufNum.Get<T>();

    AscendC::Mins(xLocal, xLocal, (T)8, calcCount);
    AscendC::Maxs(xLocal, xLocal, (T)(-8), calcCount);

    // numTensor = -x, then expNegX = exp(-x)
    AscendC::Muls(numTensor, xLocal, (T)(-1), calcCount);
    AscendC::Exp(expNegX, numTensor, calcCount);
    // expX = exp(x)
    AscendC::Exp(expX, xLocal, calcCount);
    // numTensor = exp(x) - exp(-x)
    AscendC::Sub(numTensor, expX, expNegX, calcCount);
    // expX = exp(x) + exp(-x)
    AscendC::Add(expX, expX, expNegX, calcCount);
    // y = numerator / denominator
    AscendC::Div(yLocal, numTensor, expX, calcCount);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H