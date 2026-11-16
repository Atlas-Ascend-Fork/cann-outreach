#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        this->blockLength = tilingData->blockLength;
        this->lastBlockLength = tilingData->lastBlockLength;
        this->tileNum = tilingData->tileNum;
        this->tileLength = tilingData->tileLength;
        this->lastTileLength = tilingData->lastTileLength;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t thisBlockLength = (blockIdx == AscendC::GetBlockNum() - 1)
                                       ? this->lastBlockLength
                                       : this->blockLength;
        uint32_t offset = blockIdx * this->blockLength;

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, thisBlockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, thisBlockLength);

        uint32_t initLen = (this->tileLength > this->lastTileLength)
                               ? this->tileLength
                               : this->lastTileLength;
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, initLen * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, initLen * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->tileNum > 0) {
            CopyIn(0, this->tileLength);
        }

        for (uint32_t i = 0; i < this->tileNum; i++) {
            Compute(this->tileLength);
            if (i + 1 < this->tileNum) {
                CopyIn(i + 1, this->tileLength);
            }
            CopyOut(i, this->tileLength);
        }

        if (this->lastTileLength > 0) {
            uint32_t base = this->tileNum * this->tileLength;
            CopyInPad(base, this->lastTileLength);
            Compute(this->lastTileLength);
            CopyOutPad(base, this->lastTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[idx * this->tileLength], len);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInPad(uint32_t base, uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen   = static_cast<uint32_t>(len * sizeof(DT_X));
        copyParams.srcStride  = 0;
        copyParams.dstStride  = 0;
        copyParams.rsv        = 0;

        AscendC::DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad       = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;

        AscendC::DataCopyPad(xLocal, inputGMX[base], copyParams, padParams);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();

        AscendC::Maxs(yLocal, xLocal, static_cast<DT_X>(0), len);

        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t len)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[idx * this->tileLength], yLocal, len);
        outputQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutPad(uint32_t base, uint32_t len)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen   = static_cast<uint32_t>(len * sizeof(DT_X));
        copyParams.srcStride  = 0;
        copyParams.dstStride  = 0;
        copyParams.rsv        = 0;

        AscendC::DataCopyPad(outputGMY[base], yLocal, copyParams);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength;
    uint32_t lastBlockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};

} // namespace NsRelu
#endif // RELU_H