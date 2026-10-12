/*!
 * ile relu.h
 * rief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        // 每核处理的数据长度：尾核用 lastBlockLength（可能不满 blockFactor），其余核用 blockFactor
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        this->blockLength = (blockIdx == blockNum - 1) ? tilingData->lastBlockLength : tilingData->blockFactor;
        this->tileLength = tilingData->ubFactor;
        // 核内分块数向上取整，最后一块可能不满 tileLength
        this->tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;
        this->lastTileLength = this->blockLength - (this->tileNum - 1) * this->tileLength;

        // 每个核的 GM 起始地址按标准 blockFactor 步进（前 N-1 核满块，尾核不满块），保证不重叠、不越界
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + tilingData->blockFactor * blockIdx, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + tilingData->blockFactor * blockIdx, this->blockLength);
        // 按最大块（tileLength）分配 UB，尾块 <= tileLength 安全
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t curLen = (i == loopCount - 1) ? this->lastTileLength : this->tileLength;
            CopyIn(i, curLen);
            Compute(curLen);
            CopyOut(i, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress * this->tileLength], len);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        AscendC::Relu(yLocal, xLocal, len);
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t len)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * this->tileLength], yLocal, len);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;
    uint32_t blockLength;     // 当前核实际处理的元素数
    uint32_t tileNum;         // 当前核内分块数（向上取整）
    uint32_t tileLength;      // 满块元素数
    uint32_t lastTileLength;  // 核内最后一块的实际元素数（不满块）
};

} // namespace NsRelu
#endif // RELU_H
