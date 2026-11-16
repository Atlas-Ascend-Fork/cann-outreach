/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include <cstdint>
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        this->blockLength = tilingData->blockFactor;
        this->tileLength = tilingData->ubFactor;
        
        // 计算当前核的实际数据起始偏移
        int64_t coreOffset = this->blockLength * AscendC::GetBlockIdx();
        int64_t remain = tilingData->totalNum - coreOffset;
        
        // 计算当前核实际需要处理的数据长度，防止越界
        this->currentCoreLength = remain < this->blockLength ? remain : this->blockLength;
        if (this->currentCoreLength < 0) {
            this->currentCoreLength = 0;
        }

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + coreOffset, this->currentCoreLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + coreOffset, this->currentCoreLength);

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->currentCoreLength <= 0) {
            return;
        }

        // 按照实际的 currentCoreLength 动态循环，完美处理最后一块余数
        for (int64_t offset = 0; offset < this->currentCoreLength; offset += this->tileLength) {
            int64_t processLen = this->currentCoreLength - offset;
            if (processLen > this->tileLength) {
                processLen = this->tileLength;
            }
            CopyIn(offset, processLen);
            Compute(processLen);
            CopyOut(offset, processLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t calLen)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[offset], calLen);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t calLen)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        
        // 使用官方高阶 ReLU API，天然支持任意长度（含不对齐的尾块），且高性能
        AscendC::Relu(yLocal, xLocal, calLen);
        
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t calLen)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[offset], yLocal, calLen);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    int64_t blockLength;
    int64_t tileLength;
    int64_t currentCoreLength;
};

} // namespace NsRelu
#endif // RELU_H