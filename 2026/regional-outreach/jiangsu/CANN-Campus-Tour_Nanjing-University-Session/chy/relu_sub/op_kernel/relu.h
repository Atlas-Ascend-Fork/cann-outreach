/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义：y = max(x, 0)
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
        // 多核切分：每核处理 blockLength 个元素，末核处理余数（validLength 可能更短）
        const uint32_t totalLength = tilingData->totalLength;
        const uint32_t blockLength = (totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
        const uint32_t startIdx = blockLength * AscendC::GetBlockIdx();
        this->validLength = (startIdx >= totalLength) ? 0 : (blockLength < (totalLength - startIdx) ? blockLength : (totalLength - startIdx));
        this->tileNum = tilingData->tileNum;
        this->tileLength = this->validLength / this->tileNum;
        this->remainLength = this->validLength % this->tileNum;
        // 队列按最大 tile 长度分配（最后一个 tile = tileLength + remainLength）
        this->maxTileLength = this->tileLength + this->remainLength;
        // 设置每个核的 Global Memory 起始地址（关键的多核切分逻辑）
        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + startIdx, this->validLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + startIdx, this->validLength);
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->maxTileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->maxTileLength * sizeof(DT_X));
        pipe.InitBuffer(zeroQueue, 1, this->maxTileLength * sizeof(DT_X));
        // ReLU = max(x, 0)：预生成全 0 张量，Compute 阶段与输入取逐元素最大值
        zeroLocal = zeroQueue.AllocTensor<DT_X>();
        DT_X zeroVal = 0;
        Assign(zeroLocal, zeroVal, this->maxTileLength);
    }

    __aicore__ inline void Process()
    {
        if (this->validLength == 0) {
            return;
        }
        // 共 tileNum 个 tile：前 tileNum-1 个长度为 tileLength，最后一个为 tileLength + remainLength
        // （tileLength 为 0 时，只有最后一个 tile 携带全部数据）
        for (uint32_t i = 0; i < this->tileNum; i++) {
            uint32_t curTileLength = (i == this->tileNum - 1) ? (this->tileLength + this->remainLength) : this->tileLength;
            if (curTileLength == 0) {
                continue;
            }
            uint32_t offset = i * this->tileLength;
            CopyIn(offset, curTileLength);
            Compute(curTileLength);
            CopyOut(offset, curTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, inputGMX[offset], length);
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
        Maximum(yLocal, xLocal, zeroLocal, length);
        outputQueueY.EnQue<DT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        DataCopy(outputGMY[offset], yLocal, length);
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TQue<QuePosition::VECIN, 1> zeroQueue;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;
    LocalTensor<DT_X> zeroLocal;

    uint32_t validLength;   // 本核实际处理的元素数（末核可能更短）
    uint32_t tileNum;       // 单核内分块数
    uint32_t tileLength;    // 前 tileNum-1 个 tile 的元素数
    uint32_t remainLength;  // 最后一个 tile 的额外元素数
    uint32_t maxTileLength; // 最大 tile 元素数（用于 UB 分配）
};

} // namespace NsRelu
#endif // RELU_H
