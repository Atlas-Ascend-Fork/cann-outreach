/*!
 * \file relu.h
 * \brief ReLU kernel for the supplied three-int64 tiling structure.
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {
using namespace AscendC;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        const int64_t offset = static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->blockFactor;
        if (offset >= tilingData->totalNum) {
            return;
        }
        const int64_t remaining = tilingData->totalNum - offset;
        blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        ubLength_ = tilingData->ubFactor < blockLength_ ? tilingData->ubFactor : blockLength_;

        inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
        outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);

        // Physical buffer count and queue depth are independent.
        const uint8_t bufferCount = blockLength_ <= ubLength_ ? 1 : 2;
        const uint32_t bufferBytes = static_cast<uint32_t>(ubLength_ * sizeof(T));
        pipe.InitBuffer(inputQueueX, bufferCount, bufferBytes);
        pipe.InitBuffer(outputQueueY, bufferCount, bufferBytes);
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0) {
            return;
        }
        if (blockLength_ <= ubLength_) {
            CopyIn(0, static_cast<uint32_t>(blockLength_));
            Compute(static_cast<uint32_t>(blockLength_));
            CopyOut(0, static_cast<uint32_t>(blockLength_));
            return;
        }

        const int64_t fullTiles = blockLength_ / ubLength_;
        const uint32_t tailLength = static_cast<uint32_t>(blockLength_ % ubLength_);
        const uint32_t count = static_cast<uint32_t>(ubLength_);
        int64_t offset = 0;
        for (int64_t i = 0; i < fullTiles; ++i) {
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += ubLength_;
        }
        if (tailLength != 0) {
            CopyIn(offset, tailLength);
            Compute(tailLength);
            CopyOut(offset, tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t count)
    {
        LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, inputGMX[offset], count);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
        AscendC::Relu(yLocal, xLocal, count);
        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t count)
    {
        LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
        AscendC::DataCopy(outputGMY[offset], yLocal, count);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueueX;
    TQue<QuePosition::VECOUT, 1> outputQueueY;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

} // namespace NsRelu
#endif // RELU_H
