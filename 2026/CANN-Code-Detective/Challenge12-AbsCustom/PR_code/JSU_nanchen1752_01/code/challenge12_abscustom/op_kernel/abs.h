/*!
 * \file abs.h
 * \brief Abs 算子 kernel 类定义
 */

#ifndef ABS_H
#define ABS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "abs_tiling_data.h"
#include "abs_tiling_key.h"

namespace NsAbs {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Abs {
public:
    __aicore__ inline Abs(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AbsTilingData* tilingData);
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
__aicore__ inline void Abs<T>::Init(GM_ADDR x, GM_ADDR y, const AbsTilingData* tilingData)
{
    const int64_t offset = GetBlockIdx() * tilingData->blockFactor;
    const int64_t remaining = tilingData->totalNum - offset;
    blockLength_ = remaining > tilingData->blockFactor ? tilingData->blockFactor : remaining;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Abs<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, 0};
    DataCopyPad(inputLocal, inputGMX[progress], params, padding);
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Abs<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();
    AscendC::Abs(outputLocal, inputLocal, currentNum);
    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Abs<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[progress], outputLocal, params);
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Abs<T>::Process()
{
    for (int64_t progress = 0; progress < blockLength_; progress += ubLength_) {
        const int64_t remaining = blockLength_ - progress;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsAbs
#endif // ABS_H
