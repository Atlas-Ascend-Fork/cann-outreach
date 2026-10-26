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
    TQue<QuePosition::VECCALC, 1> tmpQueue;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Exp<T>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
{
    // 本算子 TilingData 无 lastBlockLength 字段，尾核长度由 blockFactor 与 totalNum 现场推导
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t totalNum = tilingData->totalNum;

    int64_t remain = totalNum - blockIdx * blockFactor;
    if (remain < 0) {
        remain = 0;
    }
    blockLength_ = (remain < blockFactor) ? remain : blockFactor;

    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockIdx * blockFactor, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockIdx * blockFactor, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    if constexpr (IsSameType<T, half>::value) {
        // half 走 fp32 中间量，避免 half 域 Exp 溢出与精度损失
        pipe.InitBuffer(tmpQueue, 1, ubLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Exp<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
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
__aicore__ inline void Exp<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<float> tmpLocal = tmpQueue.AllocTensor<float>();
        Cast(tmpLocal, xLocal, RoundMode::CAST_NONE, currentNum);
        // 类名 Exp 会遮蔽全局指令 Exp，必须显式限定命名空间
        AscendC::Exp(tmpLocal, tmpLocal, currentNum);
        Cast(yLocal, tmpLocal, RoundMode::CAST_RINT, currentNum);
        tmpQueue.FreeTensor(tmpLocal);
    } else {
        AscendC::Exp(yLocal, xLocal, currentNum);
    }

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = blockLength_ - i * ubLength_;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsExp
#endif // EXP_H
