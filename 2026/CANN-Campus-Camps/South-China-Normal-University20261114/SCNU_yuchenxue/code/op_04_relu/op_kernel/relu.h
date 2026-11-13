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

constexpr int32_t BUFFER_NUM = RELU_BUFFER_NUM;

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        // 每核处理的数据长度（多核并行）
        this->blockLength = tilingData->blockLength;
        this->tileNum = tilingData->tileNum;
        // 单次处理的元素数（每块长度）
        this->tileLength = tilingData->tileLength;
        const uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, this->blockLength);
        // 为队列分配 UB 内存
        const uint32_t bufferCount = tileNum == 1 ? 1 : BUFFER_NUM;
        pipe.InitBuffer(inputQueueX, bufferCount, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, bufferCount, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // Small inputs take one pass; large inputs stream through two buffers.
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress * this->tileLength], this->tileLength);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        auto x = inputQueueX.DeQue<DT_X>();
        auto y = outputQueueY.AllocTensor<DT_X>();
        AscendC::Relu(y, x, this->tileLength);
        outputQueueY.EnQue(y);
        inputQueueX.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * this->tileLength], yLocal, this->tileLength);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t blockLength; // 每核处理的元素数
    uint32_t tileNum;     // 单核内分块数
    uint32_t tileLength;  // 单核内每块元素数
};

} // namespace NsRelu
#endif // RELU_H
