// Relu kernel class definition
// y = max(x, 0)
// Multi-core: each core computes its own global memory offset by blockFactor.
// Single core: split into ubLength chunks, CopyIn -> Compute -> CopyOut pipeline.
// Queue depth BUFFER_NUM = 2 means double buffer.

#pragma once

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
    __aicore__ inline Relu() {};

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
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;

    this->ubLength_ = tilingData->ubFactor;
    this->blockLength_ = blockFactor;

    // multi-core split: last core takes the remainder
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    if (blockIdx == (blockNum - 1)) {
        this->blockLength_ = totalNum - blockFactor * blockIdx;
    }
    if (this->blockLength_ < 0) {
        this->blockLength_ = 0;
    }
    // fallback when ubFactor is invalid
    if (this->ubLength_ <= 0) {
        this->ubLength_ = (this->blockLength_ > 0) ? this->blockLength_ : 1;
    }

    int64_t offset = blockFactor * blockIdx;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, this->blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    int64_t offset = progress * this->ubLength_;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(inputLocal, inputGMX[offset], copyParams, padParams);
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();
    // y = max(x, 0), one vector instruction, exact for both fp16 and fp32
    Maxs(outputLocal, inputLocal, static_cast<T>(0), static_cast<int32_t>(currentNum));
    inputQueueX.FreeTensor(inputLocal);
    outputQueueY.EnQue<T>(outputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    int64_t offset = progress * this->ubLength_;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(outputGMY[offset], outputLocal, copyParams);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // loop over ubLength chunks inside one core, last chunk takes the remainder
    int64_t loopCount = (this->blockLength_ + this->ubLength_ - 1) / this->ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = this->blockLength_ - i * this->ubLength_;
        if (currentNum > this->ubLength_) {
            currentNum = this->ubLength_;
        }
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRelu