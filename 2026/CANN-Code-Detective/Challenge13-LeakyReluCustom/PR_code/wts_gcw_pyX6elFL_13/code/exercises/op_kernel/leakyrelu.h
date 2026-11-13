/*!
 * \file leakyrelu.h
 * \brief Leakyrelu 算子 kernel 类定义
 */

#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "leakyrelu_tiling_data.h"
#include "leakyrelu_tiling_key.h"

namespace NsLeakyrelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t DATA_BLOCK_BYTES = 32;

template <typename T>
class Leakyrelu {
public:
    __aicore__ inline Leakyrelu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> floatInput;
    TBuf<QuePosition::VECCALC> floatOutput;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float negativeSlope_ = 0.0f;
};

template <typename T>
__aicore__ inline void Leakyrelu<T>::Init(GM_ADDR x, GM_ADDR y, const LeakyreluTilingData* tilingData)
{
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    if (remaining <= 0 || tilingData->ubFactor <= 0) {
        return;
    }
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    negativeSlope_ = tilingData->negativeSlope;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    if constexpr (sizeof(T) == sizeof(half)) {
        pipe.InitBuffer(floatInput, ubLength_ * sizeof(float));
        pipe.InitBuffer(floatOutput, ubLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> input = inputQueueX.AllocTensor<T>();
    const uint32_t bytes = currentNum * sizeof(T);
    if (bytes % DATA_BLOCK_BYTES == 0) {
        DataCopy(input, inputGMX[progress], static_cast<uint32_t>(currentNum));
    } else {
        const DataCopyExtParams copy{1, bytes, 0, 0, 0};
        const DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(input, inputGMX[progress], copy, pad);
    }
    inputQueueX.EnQue(input);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> input = inputQueueX.DeQue<T>();
    LocalTensor<T> output = outputQueueY.AllocTensor<T>();
    const int32_t count = static_cast<int32_t>(currentNum);
    if constexpr (sizeof(T) == sizeof(half)) {
        LocalTensor<float> x = floatInput.Get<float>();
        LocalTensor<float> y = floatOutput.Get<float>();
        Cast(x, input, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        AscendC::LeakyRelu(y, x, negativeSlope_, count);
        PipeBarrier<PIPE_V>();
        Cast(output, y, RoundMode::CAST_RINT, count);
    } else {
        AscendC::LeakyRelu(output, input, negativeSlope_, count);
    }
    outputQueueY.EnQue(output);
    inputQueueX.FreeTensor(input);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> output = outputQueueY.DeQue<T>();
    const uint32_t bytes = currentNum * sizeof(T);
    if (bytes % DATA_BLOCK_BYTES == 0) {
        DataCopy(outputGMY[progress], output, static_cast<uint32_t>(currentNum));
    } else {
        const DataCopyExtParams copy{1, bytes, 0, 0, 0};
        DataCopyPad(outputGMY[progress], output, copy);
    }
    outputQueueY.FreeTensor(output);
}

template <typename T>
__aicore__ inline void Leakyrelu<T>::Process()
{
    for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
        const int64_t remaining = blockLength_ - offset;
        const int64_t count = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(offset, count);
        Compute(count);
        CopyOut(offset, count);
    }
}

} // namespace NsLeakyrelu
#endif // LEAKYRELU_H
