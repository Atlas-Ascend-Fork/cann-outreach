#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                           const ReluTilingData* tilingData)
{
    const uint32_t blockLength =
        static_cast<uint32_t>(tilingData->blockFactor);
    const uint32_t blockOffset =
        AscendC::GetBlockIdx() * blockLength;

    tileLength = static_cast<uint32_t>(tilingData->ubFactor);
    tileCount = blockLength / tileLength;

    inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + blockOffset, blockLength);
    outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + blockOffset, blockLength);

    const uint32_t tileBytes = tileLength * sizeof(DT_X);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileBytes);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileBytes);
}

    __aicore__ inline void Process()
    {
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            const uint32_t offset = tile * tileLength;
            CopyIn(offset);
            Compute();
            CopyOut(offset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[offset], tileLength);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();

        AscendC::Relu(yLocal, xLocal, tileLength);

        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[offset], yLocal, tileLength);
        outputQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;
    uint32_t tileCount;
    uint32_t tileLength;
};

}  // namespace NsRelu

#endif  // RELU_H