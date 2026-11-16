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

    TBuf<QuePosition::VECCALC> tmpBufferExpX; // e^x
    TBuf<QuePosition::VECCALC> tmpBufferNegX; // e^(-x)
    TBuf<QuePosition::VECCALC> tmpBufferSub;  // e^x - e^(-x)

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (blockLength_ <= 0) {
        blockLength_ = 1;
    }
    if (ubLength_ <= 0) {
        ubLength_ = blockLength_;
    }

    const int64_t blockIdx = GetBlockIdx();
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockIdx * blockLength_, static_cast<uint32_t>(blockLength_));
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockIdx * blockLength_, static_cast<uint32_t>(blockLength_));

    const uint32_t ubBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubBytes);
    pipe.InitBuffer(tmpBufferExpX, ubBytes);
    pipe.InitBuffer(tmpBufferNegX, ubBytes);
    pipe.InitBuffer(tmpBufferSub, ubBytes);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<T> expXLocal = tmpBufferExpX.Get<T>();
    LocalTensor<T> expNegXLocal = tmpBufferNegX.Get<T>();
    LocalTensor<T> subLocal = tmpBufferSub.Get<T>();

    const int32_t calCount = static_cast<int32_t>(currentNum);

    // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Exp(expXLocal, xLocal, calCount);
    Muls(expNegXLocal, xLocal, (T)(-1.0f), calCount);
    Exp(expNegXLocal, expNegXLocal, calCount);
    Sub(subLocal, expXLocal, expNegXLocal, calCount);
    Add(expNegXLocal, expXLocal, expNegXLocal, calCount);
    Div(yLocal, subLocal, expNegXLocal, calCount);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (totalNum_ <= 0) {
        return;
    }

    const int64_t blockIdx = GetBlockIdx();
    const int64_t offset = blockIdx * blockLength_;
    if (offset >= totalNum_) {
        return;
    }

    int64_t num = totalNum_ - offset;
    if (num > blockLength_) {
        num = blockLength_;
    }

    const int64_t loopCount = (num + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = num - i * ubLength_;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H