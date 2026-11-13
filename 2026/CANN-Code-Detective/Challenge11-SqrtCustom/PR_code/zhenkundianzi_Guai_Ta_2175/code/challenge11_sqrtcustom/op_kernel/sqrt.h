/*!
 * \file sqrt.h
 * \brief Sqrt 算子 kernel 类定义
 */

#ifndef SQRT_H
#define SQRT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sqrt_tiling_data.h"
#include "sqrt_tiling_key.h"

namespace NsSqrt {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Sqrt {
public:
    __aicore__ inline Sqrt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData);
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

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Sqrt<T>::Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x, totalNum_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, totalNum_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen   = static_cast<uint32_t>(currentNum * sizeof(T));  // 字节数
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;

    DataCopyPadExtParams<T> padParams;
    padParams.isPad        = false;
    padParams.leftPadding  = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    int32_t calcNum = static_cast<int32_t>(currentNum);
    AscendC::Sqrt(yLocal, xLocal, calcNum);

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen   = static_cast<uint32_t>(currentNum * sizeof(T));  // 字节数
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;

    DataCopyPad(outputGMY[progress], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Process()
{
    if (totalNum_ == 0 || ubLength_ <= 0) {
        return;
    }

    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t offset = blockIdx * blockLength_;
    if (offset >= totalNum_) {
        return;
    }

    int64_t curBlockLen = totalNum_ - offset;
    if (curBlockLen > blockLength_) {
        curBlockLen = blockLength_;
    }

    for (int64_t i = 0; i < curBlockLen; i += ubLength_) {
        int64_t currentNum = curBlockLen - i;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        CopyIn(offset + i, currentNum);
        Compute(currentNum);
        CopyOut(offset + i, currentNum);
    }
}

} // namespace NsSqrt
#endif // SQRT_H
