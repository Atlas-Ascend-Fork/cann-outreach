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

constexpr int32_t BUFFER_NUM = 2; // 每队列 buffer 块数（双缓冲）
constexpr uint32_t SMALL_SHAPE_MAX = 1000000;

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        this->totalLength = tilingData->totalLength;
        uint32_t blkIdx = AscendC::GetBlockIdx();
        if (this->totalLength <= SMALL_SHAPE_MAX) {
            // ===== 小 shape：大核/小核模式 =====
            this->isSmall = true;
            this->tileDataNum = tilingData->tileDataNum;
            uint32_t globalBufferIndex;
            if (blkIdx < tilingData->tailBlockNum) {
                this->coreDataNum = tilingData->bigCoreDataNum;
                this->tileNum = tilingData->finalBigTileNum;
                this->tailDataNum = tilingData->bigTailDataNum;
                globalBufferIndex = tilingData->bigCoreDataNum * blkIdx;
            } else {
                this->coreDataNum = tilingData->smallCoreDataNum;
                this->tileNum = tilingData->finalSmallTileNum;
                this->tailDataNum = tilingData->smallTailDataNum;
                globalBufferIndex = tilingData->bigCoreDataNum * tilingData->tailBlockNum +
                                     tilingData->smallCoreDataNum * (blkIdx - tilingData->tailBlockNum);
            }
            inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + globalBufferIndex, this->coreDataNum);
            outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + globalBufferIndex, this->coreDataNum);
            pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
            pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        } else {
            // ===== 大 shape：整除多核流水模式 =====
            this->isSmall = false;
            uint32_t blockNum = AscendC::GetBlockNum();
            this->tileDataNum = tilingData->tileLength;
            this->tileNum = tilingData->tileNum;
            this->coreDataNum = tilingData->blockLength;
            uint32_t globalBufferIndex = this->coreDataNum * blkIdx;
            inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + globalBufferIndex, this->coreDataNum);
            outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + globalBufferIndex, this->coreDataNum);
            pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
            pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->isSmall) {
            ProcessSmall();
        } else {
            ProcessBig();
        }
    }

private:
    // 小 shape：大核/小核模式（最后一块用 tailDataNum）
    __aicore__ inline void ProcessSmall()
    {
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < this->tileNum; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i, this->processDataNum);
            Compute(this->processDataNum);
            CopyOut(i, this->processDataNum);
        }
    }

    // 大 shape：整除模式（tileNum × BUFFER_NUM 循环）
    __aicore__ inline void ProcessBig()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i, this->tileDataNum);
            Compute(this->tileDataNum);
            CopyOut(i, this->tileDataNum);
        }
    }

    __aicore__ inline void CopyIn(int32_t progress, int32_t curLen)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress * this->tileDataNum], curLen);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t curLen)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        // y = max(0, x)
        AscendC::Maxs(yLocal, xLocal, (DT_X)0, curLen);
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, int32_t curLen)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * this->tileDataNum], yLocal, curLen);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint32_t totalLength;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    bool isSmall;
};

} // namespace NsRelu
#endif // RELU_H
