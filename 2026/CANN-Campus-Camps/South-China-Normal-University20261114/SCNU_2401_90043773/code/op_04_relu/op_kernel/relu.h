/*!
 * \file relu.h
 * \brief ReLU 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        this->blockLength = tilingData->totalLength / AscendC::GetBlockNum();
        this->tileNum = tilingData->tileNum;
        
        // 单块长度（双缓冲分摊）
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 设置 Global Memory 地址偏移
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 分配 UB 队列
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0 || this->tileLength == 0) return;

        int32_t loopCount = this->tileNum * BUFFER_NUM;
        int32_t coveredLength = loopCount * this->tileLength;
        int32_t tailLength = this->blockLength - coveredLength;

        // 1. 双缓冲流水处理主数据块
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(this->tileLength);
            CopyOut(i, this->tileLength);
        }

        // 2. 处理剩余尾块
        if (tailLength > 0) {
            CopyIn(loopCount, tailLength);
            Compute(tailLength);
            CopyOut(loopCount, tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t curLen)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[progress * this->tileLength], curLen);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLen)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();

        // ReLU 计算：y = max(0, x)
        Maxs(yLocal, xLocal, static_cast<DT_X>(0.0), curLen);

        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t curLen)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * this->tileLength], yLocal, curLen);
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