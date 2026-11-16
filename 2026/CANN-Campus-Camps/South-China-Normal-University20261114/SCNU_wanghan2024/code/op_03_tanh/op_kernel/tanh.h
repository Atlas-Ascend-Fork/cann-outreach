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
    TQue<QuePosition::VECIN, BUFFER_NUM> tmpQueue;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    this->blockLength_ = tilingData->totalNum;
    this->ubLength_ = tilingData->ubFactor; // 每次处理的大小

    this->inputGMX.SetGlobalBuffer((__gm__ T*)x, this->blockLength_);
    this->outputGMY.SetGlobalBuffer((__gm__ T*)y, this->blockLength_);

    this->pipe.InitBuffer(this->inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    this->pipe.InitBuffer(this->outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = this->inputQueueX.template AllocTensor<T>();
    AscendC::DataCopy(xLocal, this->inputGMX[progress * this->ubLength_], currentNum);
    this->inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = this->inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = this->outputQueueY.template AllocTensor<T>();
    
    // 直接调用 AscendC 底层提供的 Tanh 指令（如果有的话）
    // 如果没有，这里会报错，那就需要手写 Exp 计算。
    // 昇腾底层指令库通常支持直接算。
    AscendC::Tanh(yLocal, xLocal, currentNum);

    this->outputQueueY.EnQue(yLocal);
    this->inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = this->outputQueueY.template DeQue<T>();
    AscendC::DataCopy(this->outputGMY[progress * this->ubLength_], yLocal, currentNum);
    this->outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (AscendC::GetBlockIdx() != 0) return; // 强制单核

    int64_t loopCount = this->blockLength_ / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        this->CopyIn(i, this->ubLength_);
        this->Compute(this->ubLength_);
        this->CopyOut(i, this->ubLength_);
    }
}

} // namespace NsTanh
#endif // TANH_H
