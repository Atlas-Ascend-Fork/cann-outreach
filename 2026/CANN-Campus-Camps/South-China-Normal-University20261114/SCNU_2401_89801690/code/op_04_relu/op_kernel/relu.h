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
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        totalNum_ = tilingData->totalNum;
        blockFactor_ = tilingData->blockFactor;
        ubFactor_ = tilingData->ubFactor;

        const int64_t globalOffset =
            static_cast<int64_t>(AscendC::GetBlockIdx()) * blockFactor_;

        const int64_t remaining = totalNum_ - globalOffset;
        blockLength_ =
            remaining > 0
                ? (remaining < blockFactor_ ? remaining : blockFactor_)
                : 0;

        if (blockLength_ == 0) {
            return;
        }

        inputGMX.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(x) + globalOffset,
            blockLength_);
        outputGMY.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(y) + globalOffset,
            blockLength_);

        const uint32_t tileBytes =
            static_cast<uint32_t>(ubFactor_ * sizeof(T));

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileBytes);
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileBytes);
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0) {
            return;
        }

        const int64_t loopCount =
            (blockLength_ + ubFactor_ - 1) / ubFactor_;

        for (int64_t i = 0; i < loopCount; ++i) {
            const int64_t left = blockLength_ - i * ubFactor_;
            const uint32_t currentNum =
                static_cast<uint32_t>(left < ubFactor_ ? left : ubFactor_);

            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(
        int64_t progress, uint32_t currentNum)
    {
        LocalTensor<T> inputLocal =
            inputQueueX.AllocTensor<T>();

        AscendC::DataCopy(
            inputLocal,
            inputGMX[progress * ubFactor_],
            currentNum);

        inputQueueX.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
        LocalTensor<T> outputLocal =
            outputQueueY.AllocTensor<T>();

        AscendC::Relu(
            outputLocal,
            inputLocal,
            static_cast<int32_t>(currentNum));

        outputQueueY.EnQue(outputLocal);
        inputQueueX.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(
        int64_t progress, uint32_t currentNum)
    {
        LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();

        AscendC::DataCopy(
            outputGMY[progress * ubFactor_],
            outputLocal,
            currentNum);

        outputQueueY.FreeTensor(outputLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubFactor_ = 0;
};

} // namespace NsRelu
#endif // RELU_H