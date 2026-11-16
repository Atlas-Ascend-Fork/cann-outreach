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
    __aicore__ inline Tanh() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> tanhTmpBuffer;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t totalNum = static_cast<int64_t>(tilingData->totalNum);
    int64_t blockFactor = static_cast<int64_t>(tilingData->blockFactor);
    int64_t startOffset = blockIdx * blockFactor;

    if (startOffset >= totalNum) {
        blockLength_ = 0;
        ubLength_ = 1;
        return;
    }

    blockLength_ = totalNum - startOffset;
    if (blockLength_ > blockFactor) {
        blockLength_ = blockFactor;
    }

    ubLength_ = static_cast<int64_t>(tilingData->ubFactor);
    if (ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    int64_t typeSize = static_cast<int64_t>(sizeof(T));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * typeSize);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * typeSize);

    // Tanh 临时空间：float 为 1×input，half/bf16 为 4×input；
    // 同时保证不小于一个 repeat 的最小空间（factor × 256 字节）
    int64_t tmpFactor = (typeSize == 4) ? 1 : 4;
    int64_t tmpSize = ubLength_ * typeSize * tmpFactor;
    int64_t minTmpSize = 256 * tmpFactor;
    if (tmpSize < minTmpSize) {
        tmpSize = minTmpSize;
    }
    pipe.InitBuffer(tanhTmpBuffer, tmpSize);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<uint8_t> tmpLocal = tanhTmpBuffer.Get<uint8_t>();

    AscendC::Tanh(yLocal, xLocal, tmpLocal, currentNum);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Tanh<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
        int64_t currentNum = blockLength_ - offset;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsTanh
#endif // TANH_H