/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 *
 * 计算: y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * 使用基础矢量 API (Muls / Exp / Sub / Add / Div) 组合实现，
 * 中间结果放在 VECCALC 位置的 TBuf 临时变量中。
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

    // ---- 计算过程所需临时变量（VECCALC）----
    TBuf<QuePosition::VECCALC> negXBuf;      // -x
    TBuf<QuePosition::VECCALC> expPosBuf;    // exp(x)
    TBuf<QuePosition::VECCALC> expNegBuf;    // exp(-x)
    TBuf<QuePosition::VECCALC> numeratorBuf; // exp(x) - exp(-x)
    TBuf<QuePosition::VECCALC> denomBuf;     // exp(x) + exp(-x)

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

    // 多核切分：按 blockIdx 偏移 Global Memory 起始地址，每核只处理自己那一段
    int64_t offset = blockLength_ * GetBlockIdx();
    int64_t actualLength = blockLength_;
    if (offset + actualLength > tilingData->totalNum) {
        actualLength = (tilingData->totalNum > offset) ? (tilingData->totalNum - offset) : 0;
    }
    blockLength_ = actualLength;

    inputGMX.SetGlobalBuffer((__gm__ T *)x + offset, actualLength);
    outputGMY.SetGlobalBuffer((__gm__ T *)y + offset, actualLength);

    // 双缓冲输入/输出队列
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));

    // 5 个中间结果临时 buffer
    pipe.InitBuffer(negXBuf, ubLength_ * sizeof(T));
    pipe.InitBuffer(expPosBuf, ubLength_ * sizeof(T));
    pipe.InitBuffer(expNegBuf, ubLength_ * sizeof(T));
    pipe.InitBuffer(numeratorBuf, ubLength_ * sizeof(T));
    pipe.InitBuffer(denomBuf, ubLength_ * sizeof(T));
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

    LocalTensor<T> negX = negXBuf.Get<T>();
    LocalTensor<T> expPos = expPosBuf.Get<T>();
    LocalTensor<T> expNeg = expNegBuf.Get<T>();
    LocalTensor<T> numerator = numeratorBuf.Get<T>();
    LocalTensor<T> denom = denomBuf.Get<T>();

    uint32_t count = static_cast<uint32_t>(currentNum);

    // negX = -x
    Muls(negX, xLocal, static_cast<T>(-1.0f), count);
    // expPos = exp(x), expNeg = exp(-x)
    Exp(expPos, xLocal, count);
    Exp(expNeg, negX, count);
    // numerator = exp(x) - exp(-x)
    Sub(numerator, expPos, expNeg, count);
    // denom = exp(x) + exp(-x)
    Add(denom, expPos, expNeg, count);
    // y = numerator / denom
    Div(yLocal, numerator, denom, count);

    outputQueueY.EnQue(yLocal);
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
    if (ubLength_ <= 0 || blockLength_ <= 0) {
        return;
    }
    // 完整 tile 循环
    int64_t loopCount = blockLength_ / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        CopyIn(i, ubLength_);
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }
    // 兜底：处理不足一个 tile 的尾部
    int64_t tailLength = blockLength_ - loopCount * ubLength_;
    if (tailLength > 0) {
        CopyIn(loopCount, tailLength);
        Compute(tailLength);
        CopyOut(loopCount, tailLength);
    }
}

} // namespace NsTanh
#endif // TANH_H
