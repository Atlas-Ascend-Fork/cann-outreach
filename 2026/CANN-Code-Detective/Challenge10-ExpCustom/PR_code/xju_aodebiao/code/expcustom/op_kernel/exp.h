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

    TBuf<QuePosition::VECCALC> computeBuffer;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// Each core owns an aligned, non-overlapping slice of the flat tensor.
template <typename T>
__aicore__ inline void Exp<T>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
{
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining > 0 ? (remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor) : 0;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    if constexpr (sizeof(T) == sizeof(half)) {
        pipe.InitBuffer(computeBuffer, ubLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Exp<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> input = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, 0};
    DataCopyPad(input, inputGMX[progress], copy, padding);
    inputQueueX.EnQue(input);
}

template <typename T>
__aicore__ inline void Exp<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> input = inputQueueX.DeQue<T>();
    LocalTensor<T> output = outputQueueY.AllocTensor<T>();
    if constexpr (sizeof(T) == sizeof(half)) {
        LocalTensor<float> work = computeBuffer.Get<float>();
        Cast(work, input, RoundMode::CAST_NONE, currentNum);
        AscendC::Exp(work, work, currentNum);
        Cast(output, work, RoundMode::CAST_RINT, currentNum);
    } else {
        AscendC::Exp(output, input, currentNum);
    }
    outputQueueY.EnQue(output);
    inputQueueX.FreeTensor(input);
}

template <typename T>
__aicore__ inline void Exp<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> output = outputQueueY.DeQue<T>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress], output, copy);
    outputQueueY.FreeTensor(output);
}

template <typename T>
__aicore__ inline void Exp<T>::Process()
{
    for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
        const int64_t remaining = blockLength_ - offset;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsExp
#endif // EXP_H
