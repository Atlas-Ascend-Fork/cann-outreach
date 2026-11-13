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
    
    // TODO: 为计算补齐临时变量
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueue1;
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQueue2;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t blockIdx = GetBlockIdx();
    this->blockLength_ = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockIdx * tilingData->blockFactor, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockIdx * tilingData->blockFactor, this->blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
    
    // 初始化临时变量
    pipe.InitBuffer(tmpQueue1, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpQueue2, BUFFER_NUM, this->ubLength_ * sizeof(T));
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
    LocalTensor<T> tmp1 = tmpQueue1.AllocTensor<T>();
    LocalTensor<T> tmp2 = tmpQueue2.AllocTensor<T>();

    // 极限优化：公式等价变形为 y = (e^(2x) - 1) / (e^(2x) + 1)
    // 这样只需计算 1 次 Exp，大幅减少耗时。
    // 为了防止 float16 下 e^(2x) 溢出导致 NaN，先将输入钳位在 [-5, 5]（tanh(5) 已为 1.0）
    Mins(tmp1, xLocal, static_cast<T>(5.0f), currentNum);
    Maxs(tmp1, tmp1, static_cast<T>(-5.0f), currentNum);

    // 1. tmp1 = 2 * x
    Muls(tmp1, tmp1, static_cast<T>(2.0f), currentNum);
    // 2. tmp1 = e^(2x)
    Exp(tmp1, tmp1, currentNum);
    
    // 3. tmp2 = e^(2x) - 1
    Adds(tmp2, tmp1, static_cast<T>(-1.0f), currentNum);
    // 4. tmp1 = e^(2x) + 1
    Adds(tmp1, tmp1, static_cast<T>(1.0f), currentNum);
    
    // 5. y = tmp2 / tmp1
    Div(yLocal, tmp2, tmp1, currentNum);

    tmpQueue1.FreeTensor(tmp1);
    tmpQueue2.FreeTensor(tmp2);
    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue(yLocal);
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
    int64_t loopCount = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = this->ubLength_;
        if (i == loopCount - 1) {
            currentNum = this->blockLength_ - i * this->ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H