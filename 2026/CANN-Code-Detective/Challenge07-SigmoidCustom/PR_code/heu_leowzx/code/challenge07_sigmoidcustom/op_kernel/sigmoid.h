/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class Sigmoid {
public:
    __aicore__ inline Sigmoid(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> buf, onesBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
    int64_t tileNum = 1;
    int64_t processNum = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Sigmoid<T>::Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
    int idx = GetBlockIdx(), coreNum = GetBlockNum();
    processNum = tilingData->ubFactor;
    blockLength_ = tilingData->blockFactor;
    tileLength_ = blockLength_ / BUFFER_NUM / tileNum;
    inputGMX.SetGlobalBuffer((__gm__ T*)(x)+idx*blockLength_, processNum);
    outputGMY.SetGlobalBuffer((__gm__ T*)(y)+idx*blockLength_, processNum);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, sizeof(T) * processNum);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, sizeof(T) * processNum);
    if (std::is_same_v<T,float> == false) {
        pipe.InitBuffer(buf, sizeof(float) * processNum);
    }
    pipe.InitBuffer(onesBuf, sizeof(float) * processNum);
    LocalTensor<float> ones = onesBuf.Get<float>();
    Duplicate(ones, 1.0f, processNum);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * tileLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<float> ones = onesBuf.Get<float>();
    if constexpr (std::is_same_v<T, float>) {
        LocalTensor<float> yLocal = outputQueueY.AllocTensor<float>();
        Muls(xLocal, xLocal, -1.f, currentNum);
        Exp(xLocal, xLocal, currentNum);
        Adds(xLocal, xLocal, 1.f, currentNum);
        Div(yLocal, ones, xLocal, currentNum);
        outputQueueY.EnQue(yLocal);
    } else {
        LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
        LocalTensor<float> yFloat = buf.Get<float>();
        Cast(yFloat, xLocal, RoundMode::CAST_NONE, currentNum);
        Muls(yFloat, yFloat, -1.f, currentNum);
        Exp(yFloat, yFloat, currentNum);
        Adds(yFloat, yFloat, 1.f, currentNum);
        // Reciprocal(yFloat, yFloat, currentNum);
        Div(yFloat, ones, yFloat, currentNum);
        Cast(yLocal, yFloat, RoundMode::CAST_RINT, currentNum);
        outputQueueY.EnQue(yLocal);
    }
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * tileLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
{
    // TODO: 实现 Process 逻辑
    for (int i = 0; i < BUFFER_NUM * tileNum; i++) {
        CopyIn(i, tileLength_);
        Compute(tileLength_);
        CopyOut(i, tileLength_);
    }
}

} // namespace NsSigmoid
#endif // SIGMOID_H
