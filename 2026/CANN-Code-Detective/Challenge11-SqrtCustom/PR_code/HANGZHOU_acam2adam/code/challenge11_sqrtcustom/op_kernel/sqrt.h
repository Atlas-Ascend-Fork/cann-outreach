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

    DataCopyExtParams dataCopyParams_;
    DataCopyPadExtParams<T> padParams_;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Sqrt<T>::Init(GM_ADDR x, GM_ADDR y, const SqrtTilingData* tilingData)
{
    int64_t coreId = GetBlockIdx();
    int64_t offset = coreId * tilingData->blockFactor;
    blockLength_ = tilingData->totalNum - offset;
    if (blockLength_ > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    }
    if (blockLength_ <= 0) {
        blockLength_ = 0;
        return;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);

    dataCopyParams_.blockCount = 1;
    dataCopyParams_.blockLen = 0;
    dataCopyParams_.srcStride = 0;
    dataCopyParams_.dstStride = 0;
    padParams_.isPad = false;
    padParams_.leftPadding = 0;
    padParams_.rightPadding = 0;

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    dataCopyParams_.blockLen = currentNum * sizeof(T);
    LocalTensor<T> xTmp = inputQueueX.AllocTensor<T>();
    DataCopyPad(xTmp, inputGMX[progress * ubLength_], dataCopyParams_, padParams_);
    inputQueueX.EnQue<T>(xTmp);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xTmp = inputQueueX.DeQue<T>();
    LocalTensor<T> yTmp = outputQueueY.AllocTensor<T>();
    AscendC::Sqrt<T>(yTmp, xTmp, static_cast<int32_t>(currentNum));
    inputQueueX.FreeTensor(xTmp);
    outputQueueY.EnQue<T>(yTmp);
}

template <typename T>
__aicore__ inline void Sqrt<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    dataCopyParams_.blockLen = currentNum * sizeof(T);
    LocalTensor<T> yTmp = outputQueueY.DeQue<T>();
    DataCopyPad(outputGMY[progress * ubLength_], yTmp, dataCopyParams_);
    outputQueueY.FreeTensor(yTmp);
}

template <typename T>
__aicore__ inline void Sqrt<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t totalBlock = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t progress = 0; progress < totalBlock; ++progress) {
        int64_t remain = blockLength_ - progress * ubLength_;
        int64_t currentNum = remain < ubLength_ ? remain : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsSqrt
#endif // SQRT_H
