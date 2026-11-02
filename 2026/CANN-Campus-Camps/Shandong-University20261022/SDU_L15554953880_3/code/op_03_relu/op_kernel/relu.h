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
    const int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (tilingData->blockFactor <= 0 || tilingData->ubFactor <= 0 ||
        blockOffset >= tilingData->totalNum) {
        return;
    }
    blockLength_ = tilingData->totalNum - blockOffset;
    if (blockLength_ > tilingData->blockFactor) blockLength_ = tilingData->blockFactor;

    // Also bound UB use for the standalone kernel UT, which supplies its own
    // tiling data rather than calling the host tiling function.
    constexpr int64_t blockElements = 32 / sizeof(T);
    constexpr int64_t maxTileElements = 8192;
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ > maxTileElements) ubLength_ = maxTileElements;
    ubLength_ = (ubLength_ + blockElements - 1) / blockElements * blockElements;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockOffset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    if ((currentNum * sizeof(T)) % 32 == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams params{};
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        DataCopyPadExtParams<T> pad{};
        DataCopyPad(xLocal, inputGMX[offset], params, pad);
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    const int64_t offset = progress * ubLength_;
    if ((currentNum * sizeof(T)) % 32 == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams params{};
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        DataCopyPad(outputGMY[offset], yLocal, params);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) return;
    int64_t progress = 0;
    int64_t remaining = blockLength_;
    int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
    CopyIn(progress, currentNum);
    while (remaining > 0) {
        const int64_t nextRemaining = remaining - currentNum;
        const int64_t nextNum = nextRemaining < ubLength_ ? nextRemaining : ubLength_;
        // Prefetch the next tile into the other input buffer before consuming
        // this one. Queue events order MTE2, vector ReLU and MTE3 operations.
        if (nextRemaining > 0) CopyIn(progress + 1, nextNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
        ++progress;
        remaining = nextRemaining;
        currentNum = nextNum;
    }
}

} // namespace NsRelu
#endif // RELU_H
