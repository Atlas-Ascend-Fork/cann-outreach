/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
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
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (offset >= tilingData->totalNum) {
        return;
    }
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    if (currentNum * sizeof(T) % 32 == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        const uint32_t alignElements = 32 / sizeof(T);
        const uint8_t pad = static_cast<uint8_t>(alignElements - currentNum % alignElements);
        DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padding{true, 0, pad, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], params, padding);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    if (currentNum * sizeof(T) % 32 == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, params);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }
    // Host guarantees a positive tile length aligned to 16 elements. Therefore
    // every full tile is 32-byte aligned for both supported element types.
    const int64_t tileLength = ubLength_;
    const int64_t fullTiles = blockLength_ / tileLength;
    const int64_t tailElements = blockLength_ - fullTiles * tileLength;
    if (fullTiles > 0) {
        const uint32_t count = static_cast<uint32_t>(tileLength);
        const UnaryRepeatParams repeatParams{1, 1, 8, 8};
        // All full tiles share one Counter mask. Queue operations below use
        // pipeline events and do not modify the vector mask.
        SetMaskCount();
        SetVectorMask<T, MaskMode::COUNTER>(count);
        int64_t offset = 0;
        for (int64_t progress = 0; progress < fullTiles; ++progress) {
            LocalTensor<T> xIn = inputQueueX.AllocTensor<T>();
            DataCopy(xIn, inputGMX[offset], count);
            inputQueueX.EnQue(xIn);

            LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
            LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
            // The high-dimensional overload supports external mask control.
            // Counter mode determines repeats; repeatTime=1 is a placeholder.
            AscendC::Relu<T, false>(yLocal, xLocal, MASK_PLACEHOLDER, 1, repeatParams);
            outputQueueY.EnQue(yLocal);
            inputQueueX.FreeTensor(xLocal);

            LocalTensor<T> yOut = outputQueueY.DeQue<T>();
            DataCopy(outputGMY[offset], yOut, count);
            outputQueueY.FreeTensor(yOut);
            offset += tileLength;
        }
        // Restore normal state before the count-based tail API or kernel exit.
        SetMaskNorm();
        ResetMask();
    }
    if (tailElements > 0) {
        // Only this final partial tile needs alignment checks and DataCopyPad.
        CopyIn(fullTiles, tailElements);
        Compute(tailElements);
        CopyOut(fullTiles, tailElements);
    }
}


// Independent specialization: no streaming queues or internal TPipe object.
template <typename T>
class ReluSmall {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
        const ReluTilingData* tilingData, TPipe* pipeIn)
    {
        const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
        if (offset >= tilingData->totalNum) {
            return;
        }
        const int64_t remaining = tilingData->totalNum - offset;
        blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        ubLength_ = tilingData->ubFactor;
        inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
        pipeIn->InitBuffer(singleTileBuffer, 2 * ubLength_ * sizeof(T));
    }
    __aicore__ inline void Process();

private:
    TBuf<QuePosition::VECCALC> singleTileBuffer;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void ReluSmall<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }
    LocalTensor<T> input = singleTileBuffer.Get<T>();
    LocalTensor<T> output = input[ubLength_];
    const uint32_t count = static_cast<uint32_t>(blockLength_);
    const bool aligned = count * sizeof(T) % 32 == 0;
    if (aligned) {
        DataCopy(input, inputGMX, count);
    } else {
        const uint32_t alignElements = 32 / sizeof(T);
        const uint8_t pad = static_cast<uint8_t>(alignElements - count % alignElements);
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padding{true, 0, pad, static_cast<T>(0)};
        DataCopyPad(input, inputGMX, params, padding);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Relu(output, input, count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    if (aligned) {
        DataCopy(outputGMY, output, count);
    } else {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY, output, params);
    }
    PipeBarrier<PIPE_ALL>();
}

} // namespace NsRelu
#endif // RELU_H
