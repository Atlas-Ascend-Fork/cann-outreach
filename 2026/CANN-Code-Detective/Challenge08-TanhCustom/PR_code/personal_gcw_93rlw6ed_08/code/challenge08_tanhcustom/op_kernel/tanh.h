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
    TQue<QuePosition::VECCALC, 1> tmpQueue;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t tileNum_ = 0;
    int64_t ubLength_ = 0;
    int64_t lastTileLength_ = 0;
    int64_t blockOffset_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t blockIdx = GetBlockIdx();
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    int64_t remain = totalNum - blockIdx * blockFactor;
    blockLength_ = (remain > 0) ? ((remain > blockFactor) ? blockFactor : remain) : 0;
    blockOffset_ = blockIdx * blockFactor;

    tileNum_ = (blockLength_ + ubLength_ - 1) / ubLength_;
    lastTileLength_ = blockLength_ - (tileNum_ - 1) * ubLength_;

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockOffset_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockOffset_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpQueue, 1, ubLength_ * 2 * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> calcBuf = tmpQueue.AllocTensor<float>();
    LocalTensor<float> tmp = calcBuf;
    LocalTensor<float> one = calcBuf[ubLength_];

    if constexpr (IsSameType<T, float>::value) {
        Muls(tmp, xLocal, static_cast<float>(-2.0), currentNum);
        Exp(tmp, tmp, currentNum);
        Adds(tmp, tmp, static_cast<float>(1.0), currentNum);
        Duplicate(one, static_cast<float>(1.0), currentNum);
        Div(tmp, one, tmp, currentNum);
        Muls(tmp, tmp, static_cast<float>(2.0), currentNum);
        Adds(yLocal, tmp, static_cast<float>(-1.0), currentNum);
    } else {
        Cast(tmp, xLocal, RoundMode::CAST_NONE, currentNum);
        Muls(tmp, tmp, static_cast<float>(-2.0), currentNum);
        Exp(tmp, tmp, currentNum);
        Adds(tmp, tmp, static_cast<float>(1.0), currentNum);
        Duplicate(one, static_cast<float>(1.0), currentNum);
        Div(tmp, one, tmp, currentNum);
        Muls(tmp, tmp, static_cast<float>(2.0), currentNum);
        Adds(tmp, tmp, static_cast<float>(-1.0), currentNum);
        Cast(yLocal, tmp, RoundMode::CAST_RINT, currentNum);
    }

    tmpQueue.FreeTensor(calcBuf);
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    for (int64_t i = 0; i < tileNum_; i++) {
        int64_t currentNum = (i == tileNum_ - 1) ? lastTileLength_ : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
