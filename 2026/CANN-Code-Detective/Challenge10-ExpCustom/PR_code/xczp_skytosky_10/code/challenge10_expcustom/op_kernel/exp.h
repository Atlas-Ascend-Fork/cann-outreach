/*!
 * \file exp.h
 * \brief Exp 算子 kernel 类定义
 */

#ifndef EXP_H
#define EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Exp {
public:
    __aicore__ inline Exp(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData);
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

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Exp<T>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
{
    // 多核切分：按核序号计算本核 GM 起始地址与处理长度（最后一个核处理剩余部分）
    int64_t start = static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->blockFactor;
    int64_t remain = tilingData->totalNum - start;
    this->blockLength_ = remain < tilingData->blockFactor ? remain : tilingData->blockFactor;
    // 单核内按 ubFactor 再分块，不超过本核长度
    this->ubLength_ = tilingData->ubFactor < this->blockLength_ ? tilingData->ubFactor : this->blockLength_;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, this->blockLength_);
    // 为队列分配 UB 内存
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Exp<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // 逐元素Exp（逐元素自然指数）计算：y[i] = Exp(x[i])
    AscendC::Exp(yLocal, xLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Process()
{
    // 空 tensor 或多余核直接返回，防止越权
    if (this->blockLength_ <= 0 || this->ubLength_ <= 0) {
        return;
    }
    // CopyIn -> Compute -> CopyOut 三段式流水线，尾块按实际剩余元素数处理
    int64_t loopCount = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = this->ubLength_;
        int64_t tail = this->blockLength_ - i * this->ubLength_;
        if (tail < currentNum) {
            currentNum = tail;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsExp
#endif // EXP_H
