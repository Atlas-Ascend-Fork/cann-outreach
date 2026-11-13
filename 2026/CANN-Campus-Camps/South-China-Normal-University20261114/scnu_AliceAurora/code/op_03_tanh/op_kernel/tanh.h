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

    // tanh 计算所需的中间结果缓冲（统一用 float 保证精度）
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;  // 本核实际处理的元素数
    int64_t ubLength_ = 0;     // 单次 UB 循环处理的元素数
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t blockFactor = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;

    // 按 blockIdx 计算本核在 GM 上的起点，并处理最后一个核的尾块
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t gmOffset = blockFactor * blockIdx;
    int64_t remain = tilingData->totalNum - gmOffset;
    if (remain < 0) {
        remain = 0;
    }
    this->blockLength_ = (blockFactor < remain) ? blockFactor : remain;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, this->blockLength_);

    // 输入输出队列各 BUFFER_NUM 块
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
    // 计算中间结果按 float 申请
    pipe.InitBuffer(tmpBuf1, this->ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf2, this->ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<float> f1 = tmpBuf1.Get<float>();
    LocalTensor<float> f2 = tmpBuf2.Get<float>();
    const uint32_t count = static_cast<uint32_t>(currentNum);

    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    if constexpr (sizeof(T) == sizeof(float)) {
        // float 输入: 直接计算
        Muls(f1, xLocal, 2.0f, count);
        Exp(f1, f1, count);
        Adds(f2, f1, 1.0f, count);
        Adds(f1, f1, -1.0f, count);
        Div(yLocal, f1, f2, count);
    } else {
        // half 输入: 先转 float 计算，避免 exp(2x) 溢出并保证精度
        Cast(f1, xLocal, RoundMode::CAST_NONE, count);
        Muls(f1, f1, 2.0f, count);
        Exp(f1, f1, count);
        Adds(f2, f1, 1.0f, count);
        Adds(f1, f1, -1.0f, count);
        Div(f1, f1, f2, count);
        Cast(yLocal, f1, RoundMode::CAST_RINT, count);
    }

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    int64_t progress = 0;
    while (progress < this->blockLength_) {
        int64_t currentNum = this->blockLength_ - progress;
        if (currentNum > this->ubLength_) {
            currentNum = this->ubLength_;
        }
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
        progress += currentNum;
    }
}

} // namespace NsTanh
#endif // TANH_H