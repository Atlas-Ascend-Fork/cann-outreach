/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "lib/activation/sigmoid.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Sigmoid {
public:
    __aicore__ inline Sigmoid(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData);
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

    // Sigmoid 高阶 API 临时空间（内部 fp32 计算，预留 16B/元素）
    TBuf<QuePosition::VECCALC> tmpBuf;
};

template <typename T>
__aicore__ inline void Sigmoid<T>::Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
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
    pipe.InitBuffer(tmpBuf, this->ubLength_ * 16);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // 高阶 API：内部按高精度路径计算 y = 1 / (1 + exp(-x))，half 自动提升精度
    LocalTensor<uint8_t> tmp = tmpBuf.Get<uint8_t>();
    AscendC::Sigmoid<T>(yLocal, xLocal, tmp, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
{
    // 空 tensor 或多余核直接返回，防止除零
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

} // namespace NsSigmoid
#endif // SIGMOID_H
