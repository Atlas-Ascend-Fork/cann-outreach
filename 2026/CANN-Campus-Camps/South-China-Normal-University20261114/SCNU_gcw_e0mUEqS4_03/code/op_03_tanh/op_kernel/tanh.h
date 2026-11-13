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

    // 临时 Buffer，用于存放 exp(x)、exp(-x)、中间结果
    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalNum_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    this->totalNum_ = tilingData->totalNum;
    this->blockLength_ = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;

    // 多核切分：每个核在 GM 上的起始偏移
    int64_t blockIdx = AscendC::GetBlockIdx();
    inputGMX.SetGlobalBuffer((__gm__ T*)x + this->blockLength_ * blockIdx, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + this->blockLength_ * blockIdx, this->blockLength_);

    // 输入/输出队列，BUFFER_NUM 份
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));

    // 3 个临时 Buffer，各 1 份（每轮计算复用）
    pipe.InitBuffer(tmpBuf0, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf1, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf2, this->ubLength_ * sizeof(T));
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

    LocalTensor<T> tmp0 = tmpBuf0.Get<T>();
    LocalTensor<T> tmp1 = tmpBuf1.Get<T>();
    LocalTensor<T> tmp2 = tmpBuf2.Get<T>();

    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    // 步骤 1：tmp0 = exp(x)
    AscendC::Exp(tmp0, xLocal, currentNum);

    // 步骤 2：tmp1 = -x
    AscendC::Muls(tmp1, xLocal, (T)-1.0, currentNum);

    // 步骤 3：tmp1 = exp(-x)
    AscendC::Exp(tmp1, tmp1, currentNum);

    // 步骤 4：tmp2 = exp(x) - exp(-x)  (分子)
    AscendC::Sub(tmp2, tmp0, tmp1, currentNum);

    // 步骤 5：tmp0 = exp(x) + exp(-x)  (分母，复用 tmp0)
    AscendC::Add(tmp0, tmp0, tmp1, currentNum);

    // 步骤 6：yLocal = tmp2 / tmp0
    AscendC::Div(yLocal, tmp2, tmp0, currentNum);

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
    // 本核实际需要处理的元素数（最后一个核可能少于 blockLength_）
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t coreStart = blockIdx * this->blockLength_;
    int64_t coreEnd = coreStart + this->blockLength_;
    if (coreEnd > this->totalNum_) {
        coreEnd = this->totalNum_;
    }
    int64_t actualLength = coreEnd - coreStart;
    if (actualLength <= 0) {
        return;
    }

    int64_t loopCount = (actualLength + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = this->ubLength_;
        if (i == loopCount - 1) {
            currentNum = actualLength - i * this->ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H