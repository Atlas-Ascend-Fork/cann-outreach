/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t MAX_UB_BATCH_ELE_NUM = 8192;

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
    TBuf<TPosition::VECCALC> tmpBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t alignElem_ = 8;  // 32B 对齐对应的元素数，按 dtype 在 Init 中更新
};

template <typename T>
__aicore__ inline void Sigmoid<T>::Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
{
    if (tilingData == nullptr) {
        return;
    }

    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ > MAX_UB_BATCH_ELE_NUM) {
        ubLength_ = MAX_UB_BATCH_ELE_NUM;
    }
    if (totalNum_ <= 0 || blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    alignElem_ = 32 / static_cast<int64_t>(sizeof(T));

    inputGMX.SetGlobalBuffer((__gm__ T*)x);
    outputGMY.SetGlobalBuffer((__gm__ T*)y);

    pipe.InitBuffer(tmpBuf, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // progress 为全局 GM 元素偏移（已含核基地址）；32B 对齐走 DataCopy 高速路径，非对齐尾块走 DataCopyPad
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    if (currentNum % alignElem_ == 0) {
        AscendC::DataCopy(xLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<T> tmpLocal = tmpBuf.Get<T>();

    // y = 1 / (1 + exp(-x))，全部使用无架构守卫的基础 API，兼容 CPU 仿真；
    // 除法用 Div（内部含牛顿迭代精化）而非 Reciprocal（vrec 指令仅 ~2^-8 精度，无法达标）
    const int32_t count = static_cast<int32_t>(currentNum);
    AscendC::Muls(tmpLocal, xLocal, static_cast<T>(-1.0), count);  // t = -x
    AscendC::Exp(tmpLocal, tmpLocal, count);                       // t = exp(-x)
    AscendC::Adds(tmpLocal, tmpLocal, static_cast<T>(1.0), count); // t = 1 + exp(-x)
    AscendC::Duplicate(yLocal, static_cast<T>(1.0), count);        // y = 1
    AscendC::Div(yLocal, yLocal, tmpLocal, count);                 // y = 1 / (1 + exp(-x))

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // progress 为全局 GM 元素偏移（已含核基地址）；32B 对齐走 DataCopy 高速路径，非对齐尾块走 DataCopyPad
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    if (currentNum % alignElem_ == 0) {
        AscendC::DataCopy(outputGMY[progress], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(outputGMY[progress], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t offset = blockIdx * blockLength_;
    if (offset >= totalNum_) {
        return;
    }

    int64_t currentBlockNum = totalNum_ - offset;
    if (currentBlockNum > blockLength_) {
        currentBlockNum = blockLength_;
    }
    if (currentBlockNum <= 0) {
        return;
    }

    const int64_t fullLoop = currentBlockNum / ubLength_;
    const int64_t tail = currentBlockNum % ubLength_;
    for (int64_t i = 0; i < fullLoop; ++i) {
        // progress 必须是全局 GM 偏移：核基地址 + 核内迭代偏移
        const int64_t progress = offset + i * ubLength_;
        CopyIn(progress, ubLength_);
        Compute(ubLength_);
        CopyOut(progress, ubLength_);
    }

    if (tail > 0) {
        const int64_t progress = offset + fullLoop * ubLength_;
        CopyIn(progress, tail);
        Compute(tail);
        CopyOut(progress, tail);
    }
}

} // namespace NsSigmoid
#endif // SIGMOID_H
