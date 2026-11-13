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
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // 【防御 2】向上取整，计算每核理论处理长度
        this->blockLength = (tilingData->totalLength + blockNum - 1) / blockNum;
        
        uint32_t offset = this->blockLength * blockIdx;
        if (offset >= tilingData->totalLength) {
            this->blockLength = 0; // 尾核可能分不到数据
        } else {
            uint32_t remain = tilingData->totalLength - offset;
            this->blockLength = (remain < this->blockLength) ? remain : this->blockLength;
        }

        this->tileNum = tilingData->tileNum;
        this->tileLength = (this->blockLength == 0) ? 0 : (this->blockLength / tileNum / BUFFER_NUM);
        
        // 处理极小数据量的兜底
        if (this->tileLength == 0 && this->blockLength > 0) {
            this->tileLength = this->blockLength; 
        }

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, this->blockLength);

        if (this->tileLength > 0) {
            pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
            pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) return;

        int32_t loopCount = (this->blockLength + this->tileLength - 1) / this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t currentTileLength = this->tileLength;
            if (i == loopCount - 1) {
                currentTileLength = this->blockLength - i * this->tileLength;
            }
            
            // 【防御 3】处理非 32 字节对齐的尾块
            // NPU 搬运指令要求字节数必须是 32 的倍数，如果不是，需要向上补齐
            uint32_t copyLength = currentTileLength;
            uint32_t alignBytes = 32;
            if constexpr (sizeof(DT_X) == 4) alignBytes = 8;  // float32 下 8 个元素 32 字节
            if constexpr (sizeof(DT_X) == 2) alignBytes = 16; // float16 下 16 个元素 32 字节
            
            if (copyLength % alignBytes != 0) {
                copyLength = (copyLength + alignBytes - 1) / alignBytes * alignBytes; // 向上取整对齐
            }

            CopyIn(i, copyLength);
            Compute(currentTileLength); // 计算时按实际真实长度算
            CopyOut(i, copyLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t copyLength)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        // 使用 DataCopyPad 防止越界，或者确保 copyLength 已经 32 字节对齐
        DataCopy(xLocal, inputGMX[progress * this->tileLength], copyLength);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t currentTileLength)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        
        AscendC::Relu(yLocal, xLocal, currentTileLength);
        
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t copyLength)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[progress * this->tileLength], yLocal, copyLength);
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