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
    uint32_t blockIdx = GetBlockIdx();
    int64_t gmOffset = static_cast<int64_t>(blockIdx) * tilingData->blockFactor;

    int64_t remain = tilingData->totalNum - gmOffset;
    if (remain < 0) {
        remain = 0;
    }

    blockLength_ = remain > tilingData->blockFactor ? tilingData->blockFactor : remain;
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, static_cast<uint32_t>(blockLength_));
    outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, static_cast<uint32_t>(blockLength_));

    // BUFFER_NUM = 2，实现 Double Buffer
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Abs<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    DataCopy(inputLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Abs<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();

    AscendC::Abs(outputLocal, inputLocal, static_cast<int32_t>(currentNum));

    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Abs<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress], outputLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Abs<T>::Process()
{
    int64_t offset = 0;
    while (offset < blockLength_) {
        int64_t currentNum = blockLength_ - offset;
        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }

        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);

        offset += currentNum;
    }
}
} // namespace NsAbs
#endif // ABS_H