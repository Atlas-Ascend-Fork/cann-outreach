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

    // 拼接 Tanh 所需的临时计算缓冲（UB 内）
    TBuf<TPosition::VECCALC> tmpNegBuf;     // -x
    TBuf<TPosition::VECCALC> tmpExpPosBuf;   // e^x
    TBuf<TPosition::VECCALC> tmpExpNegBuf;  // e^(-x)
    TBuf<TPosition::VECCALC> tmpNumBuf;     // 分子 e^x - e^(-x)
    TBuf<TPosition::VECCALC> tmpDenBuf;      // 分母 e^x + e^(-x)

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// Init：绑定 GM 张量、初始化 UB 队列与临时缓冲（Double Buffer）
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

    pipe.InitBuffer(tmpNegBuf, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpExpPosBuf, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpExpNegBuf, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpNumBuf, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpDenBuf, ubLength_ * sizeof(T));
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

// Compute：用基础小算子拼接 tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    int64_t alignElems = 32 / sizeof(T);
    int64_t calcCount = ((currentNum + alignElems - 1) / alignElems) * alignElems;

    AscendC::LocalTensor<T> tNeg = tmpNegBuf.Get<T>();
    AscendC::LocalTensor<T> tExpPos = tmpExpPosBuf.Get<T>();
    AscendC::LocalTensor<T> tExpNeg = tmpExpNegBuf.Get<T>();
    AscendC::LocalTensor<T> tNum = tmpNumBuf.Get<T>();
    AscendC::LocalTensor<T> tDen = tmpDenBuf.Get<T>();

    // 1) -x；2) e^x；3) e^(-x)；4) 分子；5) 分母；6) 相除得到 tanh(x)
    AscendC::Muls(tNeg, xLocal, static_cast<T>(-1.0f), calcCount);
    AscendC::Exp(tExpPos, xLocal, calcCount);
    AscendC::Exp(tExpNeg, tNeg, calcCount);
    AscendC::Sub(tNum, tExpPos, tExpNeg, calcCount);
    AscendC::Add(tDen, tExpPos, tExpNeg, calcCount);
    AscendC::Div(yLocal, tNum, tDen, calcCount);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
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
