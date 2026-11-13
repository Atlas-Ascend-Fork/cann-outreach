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
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx();

        this->blockLength = totalLength / blockNum;
        if (coreIdx < (totalLength % blockNum)) {
            this->blockLength += 1;
        }

        uint32_t offset = coreIdx * (totalLength / blockNum);
        if (coreIdx < (totalLength % blockNum)) {
            offset += coreIdx;
        } else {
            offset += (totalLength % blockNum);
        }

        if (this->blockLength == 0) {
            inputGMX.SetGlobalBuffer((__gm__ DT_X*)x, 0);
            outputGMY.SetGlobalBuffer((__gm__ DT_X*)y, 0);
            return;
        }

        this->tileNum = tilingData->tileNum;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        if (this->tileLength == 0) {
            this->tileLength = 1;
        }

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, this->blockLength);

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, (this->tileLength + 16) * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, (this->tileLength + 16) * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) return;

        int32_t loopCount = (this->blockLength + this->tileLength - 1) / this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t currentNum = (i == loopCount - 1) ? (this->blockLength - i * this->tileLength) : this->tileLength;
            
            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t currentNum)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, inputGMX[progress * this->tileLength], copyParams, padParams);
        
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t currentNum)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();

        Maxs(yLocal, xLocal, (DT_X)0.0, currentNum);

        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t currentNum)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopyExtParams outParams{1, static_cast<uint32_t>(currentNum * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(outputGMY[progress * this->tileLength], yLocal, outParams);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; 
    uint32_t tileNum;     
    uint32_t tileLength;  
};

} // namespace NsRelu
#endif // RELU_H