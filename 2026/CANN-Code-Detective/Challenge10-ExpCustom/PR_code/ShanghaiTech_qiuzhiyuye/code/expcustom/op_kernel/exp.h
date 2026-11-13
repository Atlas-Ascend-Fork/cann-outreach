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

// 双缓冲用于跨 tile 隐藏部分 GM<->UB 搬运开销；单 tile 时 Init 中退化为单缓冲。
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Exp {
public:
    __aicore__ inline Exp() {};

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

template <typename T>
__aicore__ inline void Exp<T>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
{
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (offset >= tilingData->totalNum || tilingData->ubFactor <= 0) {
        return;
    }

    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    const uint8_t bufferNum = blockLength_ <= ubLength_ ? 1 : BUFFER_NUM;
    const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueX, bufferNum, bufferBytes);
    pipe.InitBuffer(outputQueueY, bufferNum, bufferBytes);
}

template <typename T>
__aicore__ inline void Exp<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    const int64_t offset = progress * ubLength_;
    constexpr int64_t alignNum = 32 / sizeof(T);

    if (currentNum % alignNum == 0) {
        AscendC::DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    const int32_t count = static_cast<int32_t>(currentNum);

    // y = exp(x)。显式使用 AscendC:: 前缀，避免与当前模板类 Exp 重名造成查找歧义。
    AscendC::Exp(yLocal, xLocal, count);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    const int64_t offset = progress * ubLength_;
    constexpr int64_t alignNum = 32 / sizeof(T);

    if (currentNum % alignNum == 0) {
        AscendC::DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        const DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Exp<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }

    if (blockLength_ <= ubLength_) {
        CopyIn(0, blockLength_);
        Compute(blockLength_);
        CopyOut(0, blockLength_);
        return;
    }

    const int64_t fullTiles = blockLength_ / ubLength_;
    const int64_t tailNum = blockLength_ % ubLength_;
    const int64_t tileCount = fullTiles + (tailNum != 0 ? 1 : 0);

    // 先预取下一 tile，再计算上一 tile，利用双缓冲让搬运与计算重叠。
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

} // namespace NsExp
#endif // EXP_H
