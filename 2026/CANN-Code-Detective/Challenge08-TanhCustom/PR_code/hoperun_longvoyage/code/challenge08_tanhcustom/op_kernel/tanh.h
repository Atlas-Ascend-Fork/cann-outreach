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
    TBuf<QuePosition::VECCALC> calcBuf;
    TBuf<QuePosition::VECCALC> numBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    LocalTensor<T> calcTmp;
    LocalTensor<T> numTensor;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // 多核切分：核 i 处理 [i*blockFactor, min((i+1)*blockFactor, totalNum))
    // 最后一核可能不满 blockFactor（totalNum 不整除 blockNum 时），用 totalNum 收尾
    int64_t start = tilingData->blockFactor * AscendC::GetBlockIdx();
    int64_t remain = tilingData->totalNum - start;
    this->blockLength_ = remain > 0 ? (remain < tilingData->blockFactor ? remain : tilingData->blockFactor) : 0;
    this->ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, this->blockLength_);
    // 为队列与计算临时缓冲分配 UB 内存
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(calcBuf, this->ubLength_ * sizeof(T));
    calcTmp = calcBuf.Get<T>(this->ubLength_);
    // 常量分子 2.0 只生成一次，循环复用（省每轮 Duplicate）
    pipe.InitBuffer(numBuf, this->ubLength_ * sizeof(T));
    numTensor = numBuf.Get<T>(this->ubLength_);
    AscendC::Duplicate(numTensor, (T)2.0f, this->ubLength_);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))，等价变形：tanh(x) = 2/(1+exp(-2x)) - 1
    // Div 三操作数不允许重叠：src0 为只读常量 2.0 缓冲，dst 为 yLocal，合法
    AscendC::Muls(calcTmp, xLocal, (T)-2.0f, currentNum);   // buf = -2x
    AscendC::Exp(calcTmp, calcTmp, currentNum);             // buf = exp(-2x)
    AscendC::Adds(calcTmp, calcTmp, (T)1.0f, currentNum);   // buf = 1+exp(-2x)
    AscendC::Div(yLocal, numTensor, calcTmp, currentNum);   // y = 2/(1+exp(-2x))，numTensor 复用
    AscendC::Adds(yLocal, yLocal, (T)-1.0f, currentNum);    // y = 2/(1+exp(-2x)) - 1
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    // 单核内按 ubFactor 分块循环，最后一块处理剩余元素
    int64_t loopCount = this->blockLength_ / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        CopyIn(i, this->ubLength_);
        Compute(this->ubLength_);
        CopyOut(i, this->ubLength_);
    }
    int64_t remain = this->blockLength_ % this->ubLength_;
    if (remain > 0) {
        CopyIn(loopCount, remain);
        Compute(remain);
        CopyOut(loopCount, remain);
    }
}

} // namespace NsTanh
#endif // TANH_H
