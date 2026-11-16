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

constexpr int32_t BUFFER_NUM = 2;  // 每队列 buffer 块数（DoubleBuffer 双缓冲）

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

    __aicore__ inline int64_t FloorAlignLocal(int64_t value, int64_t align)
    {
        return (value / align) * align;
    }

private:
    static constexpr int64_t MAX_UB_ELEMS = 2048;  // 与 host 侧一致，防止 UB 超限

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    // 计算所需临时变量：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    TBuf<TPosition::VECCALC> negXBuf;
    TBuf<TPosition::VECCALC> expXBuf;
    TBuf<TPosition::VECCALC> expNegXBuf;
    TBuf<TPosition::VECCALC> numeratorBuf;
    TBuf<TPosition::VECCALC> denominatorBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;   // 每核处理的元素数
    int64_t ubLength_ = 0;      // 单次 UB 循环处理的元素数
    int64_t alignNum_ = 1;      // 32B 对齐元素数
    int64_t blockOffset_ = 0;   // 本核 GM 起始偏移
    int64_t currentTotal_ = 0;  // 本核实际处理的元素数
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    // 本核负责的数据区间：[blockOffset_, blockOffset_ + currentTotal_)
    const int64_t totalNum = tilingData->totalNum;
    this->blockLength_ = tilingData->blockFactor;
    this->blockOffset_ = static_cast<int64_t>(AscendC::GetBlockIdx()) * this->blockLength_;
    if (this->blockOffset_ >= totalNum) {
        this->currentTotal_ = 0;
    } else {
        const int64_t remain = totalNum - this->blockOffset_;
        this->currentTotal_ = (remain < this->blockLength_) ? remain : this->blockLength_;
    }

    // 单次 UB 循环的元素数，向下按 32B 对齐，封顶 2048 保证 UB 不超限
    this->alignNum_ = 32 / static_cast<int64_t>(sizeof(T));
    if (this->alignNum_ < 1) {
        this->alignNum_ = 1;
    }
    this->ubLength_ = tilingData->ubFactor;
    if (this->ubLength_ <= 0 || this->ubLength_ > MAX_UB_ELEMS) {
        this->ubLength_ = MAX_UB_ELEMS;
    }
    this->ubLength_ = FloorAlignLocal(this->ubLength_, this->alignNum_);
    if (this->ubLength_ < this->alignNum_) {
        this->ubLength_ = this->alignNum_;
    }

    // 设置每个核的 Global Memory 起始地址
    inputGMX.SetGlobalBuffer((__gm__ T*)x, totalNum);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, totalNum);

    // 输入 / 输出双缓冲队列 + 5 块计算临时空间
    const uint32_t bufBytes = static_cast<uint32_t>(this->ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufBytes);
    pipe.InitBuffer(negXBuf, bufBytes);
    pipe.InitBuffer(expXBuf, bufBytes);
    pipe.InitBuffer(expNegXBuf, bufBytes);
    pipe.InitBuffer(numeratorBuf, bufBytes);
    pipe.InitBuffer(denominatorBuf, bufBytes);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[this->blockOffset_ + progress],
                      static_cast<int32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // 计算过程所需临时变量
    LocalTensor<T> negX = negXBuf.Get<T>();
    LocalTensor<T> expX = expXBuf.Get<T>();
    LocalTensor<T> expNegX = expNegXBuf.Get<T>();
    LocalTensor<T> numerator = numeratorBuf.Get<T>();
    LocalTensor<T> denominator = denominatorBuf.Get<T>();

    const int32_t count = static_cast<int32_t>(currentNum);

    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))，用 Muls/Exp/Sub/Add/Div 小算子拼接
    AscendC::Muls(negX, xLocal, static_cast<T>(-1), count);
    AscendC::Exp(expX, xLocal, count);
    AscendC::Exp(expNegX, negX, count);
    AscendC::Sub(numerator, expX, expNegX, count);
    AscendC::Add(denominator, expX, expNegX, count);
    AscendC::Div(yLocal, numerator, denominator, count);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopy(outputGMY[this->blockOffset_ + progress], yLocal,
                      static_cast<int32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (this->currentTotal_ <= 0) {
        return;
    }
    const int64_t loopCount = (this->currentTotal_ + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        const int64_t offset = i * this->ubLength_;
        int64_t currentNum = this->currentTotal_ - offset;
        if (currentNum > this->ubLength_) {
            currentNum = this->ubLength_;
        }
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H