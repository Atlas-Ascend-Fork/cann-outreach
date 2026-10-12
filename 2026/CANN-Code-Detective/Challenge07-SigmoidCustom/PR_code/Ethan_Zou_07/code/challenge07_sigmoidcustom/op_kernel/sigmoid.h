/*!
 * \file sigmoid.h
 * \brief Sigmoid 算子 kernel 类定义
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename DT_X>
class Sigmoid {
public:
    __aicore__ inline Sigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData* tilingData)
    {
        const int64_t blockOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
        const int64_t remaining = tilingData->totalNum - blockOffset;
        blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        ubLength_ = tilingData->ubFactor;
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + blockOffset, blockLength_);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + blockOffset, blockLength_);
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; ++i) {
            const int64_t remaining = blockLength_ - i * ubLength_;
            const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum)
    {
        LocalTensor<DT_X> inputLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(inputLocal, inputGMX[progress * ubLength_], currentNum);
        inputQueueX.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        LocalTensor<DT_X> inputLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> outputLocal = outputQueueY.AllocTensor<DT_X>();
        AscendC::Sigmoid<DT_X, false>(outputLocal, inputLocal, static_cast<uint32_t>(currentNum));
        inputQueueX.FreeTensor(inputLocal);
        outputQueueY.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
    {
        LocalTensor<DT_X> outputLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * ubLength_], outputLocal, currentNum);
        outputQueueY.FreeTensor(outputLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

} // namespace NsSigmoid
#endif // SIGMOID_H
