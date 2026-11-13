/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义与实现
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
    uint32_t blockIdx = GetBlockIdx();
    
    int64_t offset = blockIdx * tilingData->blockFactor;
    int64_t currentBlockLength = tilingData->totalNum - offset;
    
    if (currentBlockLength > tilingData->blockFactor) {
        currentBlockLength = tilingData->blockFactor;
    }
    
    if (currentBlockLength <= 0) {
        blockLength_ = 0;
        return;
    }

    blockLength_ = currentBlockLength;
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
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    
    DataCopyPad(xLocal, inputGMX[progress], copyParams, {false, 0, 0, 0});
    
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    Maxs(yLocal, xLocal, (T)0.0, currentNum);

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(outputGMY[progress], yLocal, copyParams);
    
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;

    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = blockLength_ - i * ubLength_;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }
        
        int64_t progress = i * ubLength_;
        
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H