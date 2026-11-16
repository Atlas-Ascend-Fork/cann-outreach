#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kTanhTmpBytes = 4096;

template <typename DT_X>
class Tanh {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               const TanhTilingData* tilingData)
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
        pipe.InitBuffer(inputQueueX, kBufferNum, tileBytes);
        pipe.InitBuffer(outputQueueY, kBufferNum, tileBytes);
        pipe.InitBuffer(tanhTmpBuf, kTanhTmpBytes);
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
        LocalTensor<uint8_t> tmp = tanhTmpBuf.Get<uint8_t>();

        AscendC::Tanh(yLocal, xLocal, tmp, tileLength);
        AscendC::PipeBarrier<PIPE_V>();

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
    TQue<QuePosition::VECIN, kBufferNum> inputQueueX;
    TQue<QuePosition::VECOUT, kBufferNum> outputQueueY;
    TBuf<TPosition::VECCALC> tanhTmpBuf;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;
    uint32_t tileCount;
    uint32_t tileLength;
};

}  // namespace NsTanh

#endif  // TANH_H