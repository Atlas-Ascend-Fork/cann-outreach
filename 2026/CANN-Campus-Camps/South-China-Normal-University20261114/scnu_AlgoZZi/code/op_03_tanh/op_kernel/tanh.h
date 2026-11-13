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
    __aicore__ inline Tanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, uint32_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, uint32_t currentNum);
    __aicore__ inline void Compute(uint32_t currentNum);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> tanhTmpBuffer;
    TBuf<TPosition::VECCALC> inputFloatBuffer;
    TBuf<TPosition::VECCALC> outputFloatBuffer;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    const int64_t offset = GetBlockIdx() * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(tanhTmpBuffer, tilingData->tmpBufferSize);
    if constexpr (sizeof(T) == sizeof(half)) {
        pipe.InitBuffer(inputFloatBuffer, ubLength_ * sizeof(float));
        pipe.InitBuffer(outputFloatBuffer, ubLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, uint32_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    if (currentNum * sizeof(T) % 32 == 0) {
        DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    } else {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(uint32_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<uint8_t> tmpLocal = tanhTmpBuffer.Get<uint8_t>();
    if constexpr (sizeof(T) == sizeof(half)) {
        LocalTensor<float> xFloat = inputFloatBuffer.Get<float>();
        LocalTensor<float> yFloat = outputFloatBuffer.Get<float>();
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, currentNum);
        PipeBarrier<PIPE_V>();
        AscendC::Tanh<float>(yFloat, xFloat, tmpLocal, currentNum);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, yFloat, RoundMode::CAST_RINT, currentNum);
    } else {
        AscendC::Tanh<T>(yLocal, xLocal, tmpLocal, currentNum);
    }
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, uint32_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    if (currentNum * sizeof(T) % 32 == 0) {
        DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    } else {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    CopyIn(0, static_cast<uint32_t>(blockLength_ < ubLength_ ? blockLength_ : ubLength_));
    for (int64_t i = 0; i < loopCount; ++i) {
        if (i + 1 < loopCount) {
            const int64_t nextRemaining = blockLength_ - (i + 1) * ubLength_;
            CopyIn(i + 1, static_cast<uint32_t>(nextRemaining < ubLength_ ? nextRemaining : ubLength_));
        }
        const int64_t remaining = blockLength_ - i * ubLength_;
        const uint32_t currentNum = static_cast<uint32_t>(remaining < ubLength_ ? remaining : ubLength_);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H
