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

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        uint32_t totalLength = tilingData->totalLength;
        uint32_t tileNum = tilingData->tileNum;
        if (tileNum == 0) {
            tileNum = 1;
        }
        this->blockNum = AscendC::GetBlockNum();
        if (this->blockNum == 0) {
            this->blockNum = 1;
        }
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t base = totalLength / this->blockNum;
        uint32_t rem = totalLength % this->blockNum;
        this->ubLength = base + (blockIdx < rem ? 1u : 0u);
        uint32_t startOffset = blockIdx * base + (blockIdx < rem ? blockIdx : rem);

        this->tileLength = this->ubLength / tileNum / BUFFER_NUM;
        if (this->tileLength == 0) {
            this->tileLength = 1;
        }

        uint32_t bufLength = this->ubLength > 0 ? this->ubLength : 1;
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + startOffset, bufLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + startOffset, bufLength);

        uint32_t dataBytes = this->tileLength * static_cast<uint32_t>(sizeof(DT_X));
        dataBytes = (dataBytes + 31u) / 32u * 32u;

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, dataBytes);
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, dataBytes);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t progress = 0; progress < this->ubLength; progress += this->tileLength) {
            uint32_t count = this->ubLength - progress;
            if (count > this->tileLength) {
                count = this->tileLength;
            }
            CopyIn(progress, count);
            Compute(count);
            CopyOut(progress, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress], count);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        AscendC::Relu(yLocal, xLocal, count);
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t count)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress], yLocal, count);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockNum;
    uint32_t ubLength;
    uint32_t tileLength;
};

} // namespace NsRelu
#endif // RELU_H
