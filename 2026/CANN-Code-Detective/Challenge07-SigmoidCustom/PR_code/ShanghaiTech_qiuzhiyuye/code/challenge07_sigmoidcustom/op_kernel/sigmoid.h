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

// 双缓冲用于跨 tile 隐藏部分 GM<->UB 搬运开销；单 tile 时 Init 中会自动退化为单缓冲。
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Sigmoid {
public:
    __aicore__ inline Sigmoid() {};

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

    // 当前核负责的元素数，以及一次 UB tile 可容纳的元素数。
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Sigmoid<T>::Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
{
    // 每个 block 处理一段连续区间。host 侧将 blockFactor 对齐到 32B，
    // 因而不同 block 的 GM 起始地址始终满足 DMA 对齐要求。
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (offset >= tilingData->totalNum || tilingData->ubFactor <= 0) {
        return;
    }

    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    // 只有确实需要多个 tile 时才启用双缓冲，减少固定形状小算子的 UB 占用。
    const uint8_t bufferNum = blockLength_ <= ubLength_ ? 1 : BUFFER_NUM;
    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX, bufferNum, bufferBytes);
    pipe.InitBuffer(outputQueueY, bufferNum, bufferBytes);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    constexpr int64_t alignNum = 32 / sizeof(T);

    // 对齐 tile 走更轻量的 DataCopy；尾 tile 只搬有效字节，避免越界读取。
    if (currentNum % alignNum == 0) {
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    const int32_t count = static_cast<int32_t>(currentNum);

    // y = 1 / (1 + exp(-x)).
    // xLocal 作为中间结果复用输入 UB，yLocal 只保存最终结果，
    // 无需额外 workspace，也避免了 UB 中的第三份大 buffer。
    Muls(xLocal, xLocal, static_cast<T>(-1), count);
    Exp(yLocal, xLocal, count);
    Adds(yLocal, yLocal, static_cast<T>(1), count);
    // 使用 Div 而不是 Reciprocal，避免取倒数近似带来的精度损失。
    Duplicate(xLocal, static_cast<T>(1), count);
    Div(yLocal, xLocal, yLocal, count);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    constexpr int64_t alignNum = 32 / sizeof(T);

    if (currentNum % alignNum == 0) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }

    // 常见场景下一个 block 的全部数据可放入 UB，避免无意义的循环控制。
    if (blockLength_ <= ubLength_) {
        CopyIn(0, blockLength_);
        Compute(blockLength_);
        CopyOut(0, blockLength_);
        return;
    }

    const int64_t fullTiles = blockLength_ / ubLength_;
    const int64_t tailNum = blockLength_ % ubLength_;
    const int64_t tileCount = fullTiles + (tailNum != 0 ? 1 : 0);

    // 先预取下一 tile，再计算上一 tile，利用 VECIN/VECOUT 双缓冲让搬运与计算重叠。
    CopyIn(0, ubLength_);
    for (int64_t i = 1; i < tileCount; ++i) {
        const int64_t currentNum = i < fullTiles ? ubLength_ : tailNum;
        const int64_t previousNum = (i - 1) < fullTiles ? ubLength_ : tailNum;
        CopyIn(i, currentNum);
        Compute(previousNum);
        CopyOut(i - 1, previousNum);
    }

    const int64_t lastNum = (tileCount - 1) < fullTiles ? ubLength_ : tailNum;
    Compute(lastNum);
    CopyOut(tileCount - 1, lastNum);
}

} // namespace NsSigmoid
#endif // SIGMOID_H



