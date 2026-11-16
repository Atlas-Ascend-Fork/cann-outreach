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

    // half 场景下用于 float32 中间计算的临时 buffer
    TBuf<TPosition::VECCALC> tmpBufA;
    TBuf<TPosition::VECCALC> tmpBufB;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    int64_t offset = blockIdx * blockFactor;

    blockLength_ = 0;
    ubLength_ = 0;
    if (offset >= totalNum || blockFactor <= 0) {
        return;
    }
    int64_t leftNum = totalNum - offset;
    blockLength_ = leftNum < blockFactor ? leftNum : blockFactor;
    ubLength_ = tilingData->ubFactor < blockLength_ ? tilingData->ubFactor : blockLength_;
    if (ubLength_ <= 0) {
        blockLength_ = 0;
        return;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);

    // TQue 的 buffer 大小需按 32 字节对齐
    constexpr int64_t UB_BLOCK_NUM = 32 / sizeof(T);
    int64_t bufferLength = ((ubLength_ + UB_BLOCK_NUM - 1) / UB_BLOCK_NUM) * UB_BLOCK_NUM;
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferLength * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferLength * sizeof(T));
    if constexpr (sizeof(T) == 2) {
        // 输入、输出各 BUFFER_NUM 块 half，加上 2 块 float32 临时空间，共 16 字节/元素
        pipe.InitBuffer(tmpBufA, bufferLength * sizeof(float));
        pipe.InitBuffer(tmpBufB, bufferLength * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    int32_t count = static_cast<int32_t>(currentNum);
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();

    // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    if constexpr (sizeof(T) == 2) {
        // half 在 float32 下计算，避免 exp 精度损失
        LocalTensor<float> xFp32 = tmpBufA.Get<float>();
        LocalTensor<float> yFp32 = tmpBufB.Get<float>();
        Cast(xFp32, inputLocal, RoundMode::CAST_NONE, count);
        Muls(xFp32, xFp32, 2.0f, count);
        Exp(xFp32, xFp32, count);
        Adds(yFp32, xFp32, 1.0f, count);
        Adds(xFp32, xFp32, -1.0f, count);
        Div(xFp32, xFp32, yFp32, count);
        Cast(outputLocal, xFp32, RoundMode::CAST_RINT, count);
    } else {
        Muls(outputLocal, inputLocal, 2.0f, count);
        Exp(outputLocal, outputLocal, count);
        Adds(inputLocal, outputLocal, 1.0f, count);
        Adds(outputLocal, outputLocal, -1.0f, count);
        Div(outputLocal, outputLocal, inputLocal, count);
    }

    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], outputLocal, copyParams);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
