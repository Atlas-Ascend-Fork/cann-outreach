#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

// v3: 1 buffer per queue. Host doubles ubFactor accordingly, so each core
//     issues half as many tiles. Prefetch loop must be removed: with queue
//     depth 1, AllocTensor for tile i+1 would deadlock before DeQue of tile i.
constexpr int32_t BUFFER_NUM = 1;

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
    __aicore__ inline int64_t CurrentNum(int64_t progress);
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t coreLength_ = 0;
};

template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) { ubLength_ = 1; }

    int64_t offset = GetBlockIdx() * blockLength_;
    if (offset >= totalNum_) { coreLength_ = 0; return; }
    int64_t rest = totalNum_ - offset;
    coreLength_ = rest < blockLength_ ? rest : blockLength_;
    if (coreLength_ <= 0) { return; }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, coreLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, coreLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline int64_t Relu<T>::CurrentNum(int64_t progress)
{
    int64_t done = progress * ubLength_;
    int64_t rest = coreLength_ - done;
    return rest < ubLength_ ? rest : ubLength_;
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (coreLength_ <= 0) { return; }
    int64_t loopCount = (coreLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        CopyIn(i, CurrentNum(i));
        Compute(CurrentNum(i));
        CopyOut(i, CurrentNum(i));
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
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
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

} // namespace NsRelu
#endif // RELU_H
