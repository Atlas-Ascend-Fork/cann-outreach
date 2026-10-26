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

constexpr int32_t BUFFER_NUM = 1;

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

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Exp<T>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
{
      // 当前核的全局偏移与负责的数据长度（尾核截断）
    int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    int64_t remain = tilingData->totalNum - blockOffset;
    if (remain < 0) {
        remain = 0;
    }
    blockLength_ = remain < tilingData->blockFactor ? remain : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockOffset, blockLength_);

    if (ubLength_ > 0) {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void Exp<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    // 使用 DataCopyPad 支持尾部非 32B 对齐的数据长度
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // y[i] = exp(x[i])，注意与类名 Exp 区分，需带 AscendC 命名空间
    AscendC::Exp(yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Process()
{
    if (ubLength_ <= 0) {
        return;
    }

    // tile 数 = ceil(blockLength_ / ubLength_)，最后一个 tile 可能不足
    int64_t tileNum = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t progress = 0; progress < tileNum; ++progress) {
        int64_t remain = blockLength_ - progress * ubLength_;
        int64_t currentNum = remain < ubLength_ ? remain : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsExp
#endif // EXP_H
