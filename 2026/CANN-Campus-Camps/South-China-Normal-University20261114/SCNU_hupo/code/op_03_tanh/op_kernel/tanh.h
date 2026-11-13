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
    TQue<QuePosition::VECCALC, 1> tmpQueue;  // 计算中间变量（fp32，两块）

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;  // 本核处理的元素数（尾核取剩余部分）
    int64_t ubLength_ = 0;     // UB 单次循环处理的元素数
};

// Init：绑定 GM 张量并初始化 UB 队列（Double Buffer）
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    // 中间计算变量两块 fp32（tmpA / tmpB），各 ubLength_ 元素
    pipe.InitBuffer(tmpQueue, 1, ubLength_ * 2 * static_cast<int64_t>(sizeof(float)));
}

// CopyIn：把当前 tile 的 x 从 GM 搬入 UB 并入队
template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

// Compute：y = tanh(x) = (e^(2x)-1)/(e^(2x)+1)
//   中间过程全用 fp32；标量 API（Muls/Adds）+ 2 块 tmp，指令最少
template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    int64_t alignElems = 32 / sizeof(T);
    int64_t calcCount64 = ((currentNum + alignElems - 1) / alignElems) * alignElems;
    int32_t calcCount = static_cast<int32_t>(calcCount64);
    int32_t halfTmp = ubLength_;
    AscendC::LocalTensor<float> tmpA = tmpQueue.AllocTensor<float>();
    AscendC::LocalTensor<float> tmpB = tmpA[halfTmp];

    // tmpA = (float)x
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Muls(tmpA, xLocal, static_cast<float>(1.0), calcCount);
    } else {
        AscendC::Cast(tmpA, xLocal, RoundMode::CAST_NONE, calcCount);
    }
    // tmpB = 2*x = x+x
    AscendC::Add(tmpB, tmpA, tmpA, calcCount);
    // tmpA = e^(2x)
    AscendC::Exp(tmpA, tmpB, calcCount);
    // tmpB = e^(2x) + 1（分母）
    AscendC::Adds(tmpB, tmpA, static_cast<float>(1.0), calcCount);
    // tmpA = e^(2x) - 1（分子）
    AscendC::Adds(tmpA, tmpA, static_cast<float>(-1.0), calcCount);
    // tmpA = (e^(2x)-1)/(e^(2x)+1) = tanh(x)
    AscendC::Div(tmpA, tmpA, tmpB, calcCount);
    // y = (T)tanh(x)
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Muls(yLocal, tmpA, static_cast<float>(1.0), calcCount);
    } else {
        AscendC::Cast(yLocal, tmpA, RoundMode::CAST_NONE, calcCount);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
    tmpQueue.FreeTensor(tmpA);
}

// CopyOut：出队结果，从 UB 回拷 GM（只搬有效部分 currentNum）
template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

// Process：按 ubLength_ 切块循环，CopyIn → Compute → CopyOut
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
