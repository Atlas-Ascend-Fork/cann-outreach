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

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        // 多核切分：按 32B 对齐的块进行分配，避免 GM 起始地址非对齐；仅最后一个核附带元素尾部
        uint32_t alignElem = 32 / sizeof(DT_X);
        if (alignElem == 0) {
            alignElem = 1;
        }
        uint32_t blockNum = AscendC::GetBlockNum();
        if (blockNum == 0) {
            blockNum = 1;
        }
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockNum) {
            blockIdx = blockNum - 1;
        }
        uint32_t totalLength = tilingData->totalLength;
        uint32_t totalBlocks = totalLength / alignElem;
        uint32_t tailLength = totalLength - totalBlocks * alignElem;
        uint32_t baseBlocks = totalBlocks / blockNum;
        uint32_t remainBlocks = totalBlocks % blockNum;
        uint32_t myBlocks = baseBlocks + (blockIdx < remainBlocks ? 1U : 0U);
        uint32_t startBlock = blockIdx * baseBlocks + (blockIdx < remainBlocks ? blockIdx : remainBlocks);

        this->blockLength = myBlocks * alignElem;
        uint32_t gmOffset = startBlock * alignElem;
        if (totalBlocks == 0) {
            // 数据量不足一个对齐块时，由 0 号核处理全部数据
            if (blockIdx == 0) {
                this->blockLength = tailLength;
            }
        } else if (startBlock + myBlocks == totalBlocks) {
            // 最后一个对齐块所属的核负责处理元素尾部
            this->blockLength += tailLength;
        }

        this->tileNum = tilingData->tileNum;
        uint32_t loopNum = this->tileNum * BUFFER_NUM;
        if (loopNum == 0) {
            loopNum = BUFFER_NUM;
        }
        // 单次处理的元素数（向上取整，并按 32B 对齐，首尾均可用 DataCopyPad 安全处理）
        this->tileLength = (this->blockLength + loopNum - 1) / loopNum;
        this->tileLength = (this->tileLength + alignElem - 1) / alignElem * alignElem;
        if (this->tileLength == 0) {
            this->tileLength = alignElem;
        }

        // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + gmOffset, this->blockLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + gmOffset, this->blockLength);
        // 为队列分配 UB 内存
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 按块循环，最后一块可能不足 tileLength，由 DataCopyPad 按实际长度搬运
        uint32_t offset = 0;
        while (offset < this->blockLength) {
            uint32_t count = this->blockLength - offset;
            if (count > this->tileLength) {
                count = this->tileLength;
            }
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopyExtParams copyParams(1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0);
        DataCopyPadExtParams<DT_X> padParams(false, 0, 0, static_cast<DT_X>(0));
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        // Relu(x) = max(x, 0)，逐元素计算
        Maxs(yLocal, xLocal, static_cast<DT_X>(0), count);
        inputQueueX.FreeTensor(xLocal);
        outputQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopyExtParams copyParams(1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0);
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
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
