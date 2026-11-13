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
constexpr int64_t BLOCK_BYTES = 32;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu() {};

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData);

    __aicore__ inline void Process();

private:
    // progress 表示相对于当前核 GM 起点的元素偏移。
    __aicore__ inline void CopyIn(
        int64_t progress, int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t progress, int64_t currentNum);

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
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    ubLength_ = tilingData->ubFactor;

    const int64_t blockOffset =
        static_cast<int64_t>(GetBlockIdx()) *
        tilingData->blockFactor;

    if (blockOffset >= tilingData->totalNum) {
        blockLength_ = 0;
        return;
    }

    const int64_t remaining =
        tilingData->totalNum - blockOffset;

    blockLength_ =
        (remaining < tilingData->blockFactor)
            ? remaining
            : tilingData->blockFactor;

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + blockOffset, blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + blockOffset, blockLength_);

    const uint32_t bufferBytes =
        static_cast<uint32_t>(ubLength_ * sizeof(T));

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, bufferBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, bufferBytes);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    const uint32_t copyBytes =
        static_cast<uint32_t>(currentNum * sizeof(T));

    if (copyBytes % BLOCK_BYTES == 0) {
        DataCopy(
            xLocal,
            inputGMX[progress],
            static_cast<uint32_t>(currentNum));
    } else {
        // 尾块仅搬运有效元素，在 UB 中补齐至 32 字节。
        DataCopyExtParams copyParams{
            1, copyBytes, 0, 0, 0
        };

        const uint8_t rightPadding =
            static_cast<uint8_t>(
                (BLOCK_BYTES - copyBytes % BLOCK_BYTES) /
                sizeof(T));

        DataCopyPadExtParams<T> padParams{
            true, 0, rightPadding, static_cast<T>(0)
        };

        DataCopyPad(
            xLocal,
            inputGMX[progress],
            copyParams,
            padParams);
    }

    inputQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    AscendC::Relu(
        yLocal,
        xLocal,
        static_cast<int32_t>(currentNum));

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    const uint32_t copyBytes =
        static_cast<uint32_t>(currentNum * sizeof(T));

    if (copyBytes % BLOCK_BYTES == 0) {
        DataCopy(
            outputGMY[progress],
            yLocal,
            static_cast<uint32_t>(currentNum));
    } else {
        // 写回真实字节数，避免尾块覆盖输出张量之外的内存。
        DataCopyExtParams copyParams{
            1, copyBytes, 0, 0, 0
        };

        DataCopyPad(
            outputGMY[progress],
            yLocal,
            copyParams);
    }

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    for (int64_t progress = 0;
         progress < blockLength_;
         progress += ubLength_) {

        const int64_t remaining = blockLength_ - progress;

        const int64_t currentNum =
            (remaining < ubLength_) ? remaining : ubLength_;

        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu

#endif // RELU_H