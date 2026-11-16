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
    TBuf<QuePosition::VECCALC> smallBuffer;
    bool smallMode_ = false;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// GM -> UB -> elementwise ReLU -> GM, with matched queue lifetimes.
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    smallMode_ = tilingData->totalNum == 16384;
    if (smallMode_) {
        pipe.InitBuffer(smallBuffer, ubLength_ * sizeof(T));
    } else {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, currentNum);
    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    if (smallMode_) {
        // In-place UB ReLU; explicit events preserve every data dependency.
        LocalTensor<T> local = smallBuffer.Get<T>();
        const int32_t eLoad = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        const int32_t eStore = static_cast<int32_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        const int32_t eReuse = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        for (int64_t offset = 0; offset < blockLength_; offset += ubLength_) {
            const int64_t remaining = blockLength_ - offset;
            const int64_t count = remaining < ubLength_ ? remaining : ubLength_;
            DataCopy(local, inputGMX[offset], count);
            SetFlag<HardEvent::MTE2_V>(eLoad);
            WaitFlag<HardEvent::MTE2_V>(eLoad);
            AscendC::Relu(local, local, count);
            SetFlag<HardEvent::V_MTE3>(eStore);
            WaitFlag<HardEvent::V_MTE3>(eStore);
            DataCopy(outputGMY[offset], local, count);
            SetFlag<HardEvent::MTE3_MTE2>(eReuse);
            WaitFlag<HardEvent::MTE3_MTE2>(eReuse);
        }
        return;
    }
    // Prime the pipeline, then enqueue tile i+1 before computing tile i.
    const int64_t firstCount = blockLength_ < ubLength_ ? blockLength_ : ubLength_;
    CopyIn(0, firstCount);
    for (int64_t progress = 0; progress < loopCount; ++progress) {
        const int64_t remaining = blockLength_ - progress * ubLength_;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        if (progress + 1 < loopCount) {
            const int64_t nextRemaining = blockLength_ - (progress + 1) * ubLength_;
            const int64_t nextNum = nextRemaining < ubLength_ ? nextRemaining : ubLength_;
            CopyIn(progress + 1, nextNum);
        }
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsRelu
#endif // RELU_H